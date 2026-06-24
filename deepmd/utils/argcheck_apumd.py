# SPDX-License-Identifier: LGPL-3.0-or-later
from dargs import (
    Argument,
)


def apumd_args(fold_subdoc: bool = False) -> Argument:
    doc_version = (
        "configuration the apumd version; APUMD training currently supports the fixed value 1"
    )
    doc_max_nnei = "configuration the max number of neighbors; APUMD training currently supports the fixed value 256"
    doc_device = "hardware used by model; APUMD training currently supports the fixed value vu13p"
    doc_net_size_file = (
        "configuration the number of nodes of fitting_net; APUMD training currently supports the fixed value 128"
    )
    doc_map_file = "A file containing the mapping tables to replace the calculation of embedding nets"
    doc_config_file = "A file containing the parameters about how to implement the model in certain hardware"
    doc_weight_file = "a *.npy file containing the weights of the model"
    doc_enable = "enable the apumd training"
    doc_restore_descriptor = (
        "enable to restore the parameter of embedding_net from weight.npy"
    )
    doc_restore_fitting_net = (
        "enable to restore the parameter of fitting_net from weight.npy"
    )
    doc_quantize_descriptor = "enable the quantizatioin of descriptor"
    doc_quantize_fitting_net = "enable the quantizatioin of fitting_net"
    args = [
        Argument("version", int, optional=False, default=1, doc=doc_version),
        Argument("device", str, optional=False, default="vu13p", doc=doc_device),
        Argument("max_nnei", int, optional=False, default=256, doc=doc_max_nnei),
        Argument("net_size", int, optional=False, default=128, doc=doc_net_size_file),
        Argument("map_file", str, optional=False, default="none", doc=doc_map_file),
        Argument(
            "config_file", str, optional=False, default="none", doc=doc_config_file
        ),
        Argument(
            "weight_file", str, optional=False, default="none", doc=doc_weight_file
        ),
        Argument("enable", bool, optional=False, default=False, doc=doc_enable),
        Argument(
            "restore_descriptor",
            bool,
            optional=False,
            default=False,
            doc=doc_restore_descriptor,
        ),
        Argument(
            "restore_fitting_net",
            bool,
            optional=False,
            default=False,
            doc=doc_restore_fitting_net,
        ),
        Argument(
            "quantize_descriptor",
            bool,
            optional=False,
            default=False,
            doc=doc_quantize_descriptor,
        ),
        Argument(
            "quantize_fitting_net",
            bool,
            optional=False,
            default=False,
            doc=doc_quantize_fitting_net,
        ),
    ]

    doc_apumd = "The apumd options."
    return Argument(
        "apumd", dict, args, [], optional=True, doc=doc_apumd, fold_subdoc=fold_subdoc
    )
