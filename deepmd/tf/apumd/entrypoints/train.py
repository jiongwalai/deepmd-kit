import copy

# SPDX-License-Identifier: LGPL-3.0-or-later
import logging
import os
from typing import (
    Any,
)

from deepmd.tf.entrypoints.freeze import (
    freeze,
)
from deepmd.tf.entrypoints.train import (
    train,
)
from deepmd.tf.env import (
    tf,
)
from deepmd.tf.apumd.data.data import (
    jdata_deepmd_input_v0,
)
from deepmd.tf.apumd.entrypoints.mapt import (
    mapt,
)
from deepmd.tf.apumd.entrypoints.wrap import (
    wrap,
)
from deepmd.tf.apumd.utils.config import (
    apumd_cfg,
)
from deepmd.tf.apumd.utils.fio import (
    FioDic,
)

log = logging.getLogger(__name__)

APUMD_FIXED_SETTINGS = {
    "version": 1,
    "device": "vu13p",
    "max_nnei": 256,
    "net_size": 128,
    "sel": 256,
}
APUMD_MAX_RCUT = 8.0
APUMD_MAX_TYPES = 31

jdata_cmd_train = {
    "INPUT": "train.json",
    "init_model": None,
    "restart": None,
    "output": "out.json",
    "init_frz_model": None,
    "mpi_log": "master",
    "log_level": 2,
    "log_path": "train.log",
    "is_compress": False,
}

jdata_cmd_freeze = {
    "checkpoint_folder": ".",
    "output": "frozen_model.pb",
    "node_names": None,
    "apumd_weight": "apumd/weight.npy",
}


def _fail_invalid_input(problem: str, suggestion: str) -> None:
    raise SystemExit(
        "Invalid APUMD training input: "
        f"{problem}\n"
        f"Suggestion: {suggestion}"
    )


def _load_validated_apumd_input(fn: str) -> dict:
    jdata = FioDic().load(fn, {})
    if "apumd" not in jdata:
        if "nvnmd" in jdata:
            _fail_invalid_input(
                "found top-level 'nvnmd' section",
                "Rename the top-level key from 'nvnmd' to 'apumd' and retry.",
            )
        _fail_invalid_input(
            "missing top-level 'apumd' section",
            "Add an 'apumd' section with rcut, rcut_smth and type_map settings.",
        )

    jdata = copy.deepcopy(jdata)
    user_apumd_input = copy.deepcopy(jdata["apumd"])
    apumd_input = copy.deepcopy(jdata_deepmd_input_v0["apumd"])
    apumd_input.update(user_apumd_input)

    for key, fixed_value in APUMD_FIXED_SETTINGS.items():
        user_value = user_apumd_input.get(key)
        if user_value is not None and user_value != fixed_value:
            _fail_invalid_input(
                f"apumd.{key}={user_value!r} is not supported",
                f"Set apumd.{key} to {fixed_value!r}.",
            )
        apumd_input[key] = fixed_value

    if "rcut" not in apumd_input:
        _fail_invalid_input(
            "missing apumd.rcut",
            f"Set apumd.rcut to a value no larger than {APUMD_MAX_RCUT}.",
        )
    if apumd_input["rcut"] > APUMD_MAX_RCUT:
        _fail_invalid_input(
            f"apumd.rcut={apumd_input['rcut']} is larger than {APUMD_MAX_RCUT}",
            f"Reduce apumd.rcut to a value <= {APUMD_MAX_RCUT}.",
        )

    if "rcut_smth" not in apumd_input:
        _fail_invalid_input(
            "missing apumd.rcut_smth",
            "Set apumd.rcut_smth to a value no larger than apumd.rcut.",
        )
    if apumd_input["rcut_smth"] > apumd_input["rcut"]:
        _fail_invalid_input(
            f"apumd.rcut_smth={apumd_input['rcut_smth']} is larger than apumd.rcut={apumd_input['rcut']}",
            "Reduce apumd.rcut_smth so that apumd.rcut_smth <= apumd.rcut.",
        )

    type_map = apumd_input.get("type_map", [])
    if len(type_map) > APUMD_MAX_TYPES:
        _fail_invalid_input(
            f"apumd.type_map contains {len(type_map)} elements",
            f"Reduce apumd.type_map to at most {APUMD_MAX_TYPES} chemical species.",
        )

    jdata["apumd"] = apumd_input
    return jdata


def normalized_input(jdata: dict, PATH_CNN: str, CONFIG_CNN: str) -> dict:
    r"""Normalize a input script file for continuous neural network."""
    f = FioDic()
    jdata = copy.deepcopy(jdata)
    # apumd
    jdata_apumd = copy.deepcopy(jdata["apumd"])
    jdata_apumd["enable"] = True
    jdata_apumd["config_file"] = CONFIG_CNN
    jdata_apumd_ = copy.deepcopy(jdata_apumd)
    # model
    jdata_model = {
        "descriptor": {
            "seed": jdata_apumd_.get("seed", 1),
            "sel": jdata_apumd_["sel"],
            "rcut": jdata_apumd_["rcut"],
            "rcut_smth": jdata_apumd_["rcut_smth"],
        },
        "fitting_net": {"seed": jdata_apumd_.get("seed", 1)},
        "type_map": [],
    }
    jdata_model["type_map"] = f.get(jdata_apumd_, "type_map", [])
    apumd_cfg.init_from_jdata(jdata_apumd)
    apumd_cfg.init_from_deepmd_input(jdata_model)
    apumd_cfg.init_train_mode("cnn")
    # training
    jdata_train = f.get(jdata, "training", {})
    jdata_train["disp_training"] = True
    jdata_train["time_training"] = True
    jdata_train["profiling"] = False
    jdata_train["disp_file"] = os.path.join(
        PATH_CNN, os.path.split(jdata_train["disp_file"])[1]
    )
    jdata_train["save_ckpt"] = os.path.join(
        PATH_CNN, os.path.split(jdata_train["save_ckpt"])[1]
    )
    #
    jdata["model"] = apumd_cfg.get_model_jdata()
    if apumd_cfg.version == 1:
        jdata["model"]["type_embedding"] = {
            "activation_function": "none",
            "use_tebd_bias": True,
        }
    jdata["apumd"] = apumd_cfg.get_apumd_jdata()
    return jdata


def normalized_input_qnn(
    jdata: dict, PATH_QNN: str, CONFIG_CNN: str, WEIGHT_CNN: str, MAP_CNN: str
) -> dict:
    r"""Normalize a input script file for quantize neural network."""
    jdata = copy.deepcopy(jdata)
    jdata_apumd = copy.deepcopy(jdata["apumd"])
    jdata_apumd["enable"] = True
    jdata_apumd["version"] = apumd_cfg.version
    jdata_apumd["device"] = apumd_cfg.device
    jdata_apumd["max_nnei"] = apumd_cfg.max_nnei
    jdata_apumd["config_file"] = CONFIG_CNN
    jdata_apumd["weight_file"] = WEIGHT_CNN
    jdata_apumd["map_file"] = MAP_CNN
    apumd_cfg.init_from_jdata(jdata_apumd)
    apumd_cfg.init_train_mode("qnn")
    jdata["apumd"] = apumd_cfg.get_apumd_jdata()
    # training
    jdata2 = jdata["training"]
    jdata2["disp_file"] = os.path.join(PATH_QNN, os.path.split(jdata2["disp_file"])[1])
    jdata2["save_ckpt"] = os.path.join(PATH_QNN, os.path.split(jdata2["save_ckpt"])[1])
    jdata2.pop("stop_batch", None)
    jdata2["numb_steps"] = 0
    jdata["training"] = jdata2
    return jdata


def train_apumd(
    *,
    INPUT: str,
    init_model: str | None,
    restart: str | None,
    init_frz_model: str | None,
    skip_neighbor_stat: bool = False,
    **kwargs: Any,
) -> None:
    # test input
    if not os.path.exists(INPUT):
        raise SystemExit(f"The input script {INPUT} does not exist")

    raw_jdata = _load_validated_apumd_input(INPUT)

    # STEP1
    PATH_CNN = "apumd_cnn"
    CONFIG_CNN = os.path.join(PATH_CNN, "config.npy")
    INPUT_CNN = os.path.join(PATH_CNN, "train.json")
    WEIGHT_CNN = os.path.join(PATH_CNN, "weight.npy")
    FRZ_MODEL_CNN = os.path.join(PATH_CNN, "frozen_model.pb")
    MAP_CNN = os.path.join(PATH_CNN, "map.npy")
    LOG_CNN = os.path.join(PATH_CNN, "train.log")
    # normalize input file
    jdata = normalized_input(raw_jdata, PATH_CNN, CONFIG_CNN)
    FioDic().save(INPUT_CNN, jdata)
    apumd_cfg.save(CONFIG_CNN)
    # train cnn
    jdata = copy.deepcopy(jdata_cmd_train)
    jdata["INPUT"] = INPUT_CNN
    jdata["log_path"] = LOG_CNN
    jdata["init_model"] = init_model
    jdata["init_frz_model"] = init_frz_model
    jdata["restart"] = restart
    jdata["skip_neighbor_stat"] = skip_neighbor_stat
    train(**jdata)
    tf.reset_default_graph()
    # freeze
    jdata = copy.deepcopy(jdata_cmd_freeze)
    jdata["checkpoint_folder"] = PATH_CNN
    jdata["output"] = FRZ_MODEL_CNN
    jdata["apumd_weight"] = WEIGHT_CNN
    freeze(**jdata)
    tf.reset_default_graph()
    # map table
    jdata = {
        "apumd_config": CONFIG_CNN,
        "apumd_weight": WEIGHT_CNN,
        "apumd_map": MAP_CNN,
    }
    mapt(**jdata)
    tf.reset_default_graph()
    # STEP2
    PATH_QNN = "apumd_qnn"
    CONFIG_QNN = os.path.join(PATH_QNN, "config.npy")
    INPUT_QNN = os.path.join(PATH_QNN, "train.json")
    WEIGHT_QNN = os.path.join(PATH_QNN, "weight.npy")
    FRZ_MODEL_QNN = os.path.join(PATH_QNN, "frozen_model.pb")
    MODEL_QNN = os.path.join(PATH_QNN, "model.pb")
    LOG_QNN = os.path.join(PATH_QNN, "train.log")

    # normalize input file
    jdata = normalized_input(raw_jdata, PATH_CNN, CONFIG_CNN)
    jdata = normalized_input_qnn(jdata, PATH_QNN, CONFIG_CNN, WEIGHT_CNN, MAP_CNN)
    FioDic().save(INPUT_QNN, jdata)
    apumd_cfg.save(CONFIG_QNN)
    # train qnn (0-step discretization)
    jdata = copy.deepcopy(jdata_cmd_train)
    jdata["INPUT"] = INPUT_QNN
    jdata["log_path"] = LOG_QNN
    jdata["skip_neighbor_stat"] = skip_neighbor_stat
    train(**jdata)
    tf.reset_default_graph()
    # freeze
    jdata = copy.deepcopy(jdata_cmd_freeze)
    jdata["checkpoint_folder"] = PATH_QNN
    jdata["output"] = FRZ_MODEL_QNN
    jdata["apumd_weight"] = WEIGHT_QNN
    freeze(**jdata)
    tf.reset_default_graph()
    # wrap
    jdata = {
        "apumd_config": CONFIG_QNN,
        "apumd_weight": WEIGHT_QNN,
        "apumd_map": MAP_CNN,
        "apumd_model": MODEL_QNN,
    }
    wrap(**jdata)
    tf.reset_default_graph()
    log.info(
        "APUMD training completed. CPU/GPU model: %s; APU model: %s",
        FRZ_MODEL_CNN,
        MODEL_QNN,
    )
