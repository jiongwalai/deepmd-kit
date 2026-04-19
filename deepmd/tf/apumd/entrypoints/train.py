# SPDX-License-Identifier: LGPL-3.0-or-later
import copy
import logging
import os
from typing import (
    Optional,
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


def normalized_input(fn, PATH_CNN, CONFIG_CNN):
    r"""Normalize a input script file for continuous neural network."""
    f = FioDic()
    jdata = f.load(fn, jdata_deepmd_input_v0)
    # apumd
    jdata_apumd = jdata_deepmd_input_v0["apumd"]
    jdata_apumd["enable"] = True
    jdata_apumd["config_file"] = CONFIG_CNN
    jdata_apumd_ = f.get(jdata, "apumd", jdata_apumd)
    jdata_apumd = f.update(jdata_apumd_, jdata_apumd)
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
    jdata["apumd"] = apumd_cfg.get_apumd_jdata()
    return jdata


def normalized_input_qnn(jdata, PATH_QNN, CONFIG_CNN, WEIGHT_CNN, MAP_CNN):
    r"""Normalize a input script file for quantize neural network."""
    #
    jdata_apumd = jdata_deepmd_input_v0["apumd"]
    jdata_apumd["enable"] = True
    jdata_apumd["version"] = apumd_cfg.version
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
    jdata["training"] = jdata2
    return jdata


def train_apumd(
    *,
    INPUT: str,
    init_model: Optional[str],
    restart: Optional[str],
    init_frz_model: Optional[str],
    skip_neighbor_stat: bool = False,
    **kwargs,
) -> None:
    # test input
    if not os.path.exists(INPUT):
        log.warning(f"The input script {INPUT} does not exist")
    # STEP1
    PATH_CNN = "apumd_cnn"
    CONFIG_CNN = os.path.join(PATH_CNN, "config.npy")
    INPUT_CNN = os.path.join(PATH_CNN, "train.json")
    WEIGHT_CNN = os.path.join(PATH_CNN, "weight.npy")
    FRZ_MODEL_CNN = os.path.join(PATH_CNN, "frozen_model.pb")
    MAP_CNN = os.path.join(PATH_CNN, "map.npy")
    LOG_CNN = os.path.join(PATH_CNN, "train.log")

    # normailize input file
    jdata_s1 = normalized_input(INPUT, PATH_CNN, CONFIG_CNN)
    FioDic().save(INPUT_CNN, jdata_s1)
    apumd_cfg.save(CONFIG_CNN)
    # train cnn
    jdata_train_s1 = copy.deepcopy(jdata_cmd_train)
    jdata_train_s1["INPUT"] = INPUT_CNN
    jdata_train_s1["log_path"] = LOG_CNN
    jdata_train_s1["init_model"] = init_model
    jdata_train_s1["init_frz_model"] = init_frz_model
    jdata_train_s1["restart"] = restart
    jdata_train_s1["skip_neighbor_stat"] = skip_neighbor_stat
    train(**jdata_train_s1)
    tf.reset_default_graph()
    # freeze
    jdata_freeze_s1 = copy.deepcopy(jdata_cmd_freeze)
    jdata_freeze_s1["checkpoint_folder"] = PATH_CNN
    jdata_freeze_s1["output"] = FRZ_MODEL_CNN
    jdata_freeze_s1["apumd_weight"] = WEIGHT_CNN
    freeze(**jdata_freeze_s1)
    tf.reset_default_graph()
    # map table
    jdata_map_s1 = {
        "apumd_config": CONFIG_CNN,
        "apumd_weight": WEIGHT_CNN,
        "apumd_map": MAP_CNN,
    }
    mapt(**jdata_map_s1)
    tf.reset_default_graph()
    
    # STEP2
    PATH_QNN = "apumd_qnn"
    CONFIG_QNN = os.path.join(PATH_QNN, "config.npy")
    INPUT_QNN = os.path.join(PATH_QNN, "train.json")
    WEIGHT_QNN = os.path.join(PATH_QNN, "weight.npy")
    FRZ_MODEL_QNN = os.path.join(PATH_QNN, "frozen_model.pb")
    MODEL_QNN = os.path.join(PATH_QNN, "model.pb")
    LOG_QNN = os.path.join(PATH_QNN, "train.log")


    # normailize input file
    jdata_s2 = normalized_input(INPUT, PATH_CNN, CONFIG_CNN)
    jdata_s2 = normalized_input_qnn(jdata_s1, PATH_QNN, CONFIG_CNN, WEIGHT_CNN, MAP_CNN)
    jdata_s2["training"]["numb_steps"] = 0
    FioDic().save(INPUT_QNN, jdata_s2)
    apumd_cfg.save(CONFIG_QNN)
    # train qnn
    jdata_train_s2 = copy.deepcopy(jdata_cmd_train)
    jdata_train_s2["INPUT"] = INPUT_QNN
    jdata_train_s2["log_path"] = LOG_QNN
    jdata_train_s2["skip_neighbor_stat"] = skip_neighbor_stat
    train(**jdata_train_s2)
    tf.reset_default_graph()
    # freeze
    jdata_freeze_s2 = copy.deepcopy(jdata_cmd_freeze)
    jdata_freeze_s2["checkpoint_folder"] = PATH_QNN
    jdata_freeze_s2["output"] = FRZ_MODEL_QNN
    jdata_freeze_s2["apumd_weight"] = WEIGHT_QNN
    freeze(**jdata_freeze_s2)
    tf.reset_default_graph()
    # wrap
    jdata_wrap_s2 = {
        "apumd_config": CONFIG_QNN,
        "apumd_weight": WEIGHT_QNN,
        "apumd_map": MAP_CNN,
        "apumd_model": MODEL_QNN,
    }
    wrap(**jdata_wrap_s2)
    tf.reset_default_graph()

    log.info("APUMD training completed: s1 → s2 pipeline finished.")