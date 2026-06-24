# SPDX-License-Identifier: LGPL-3.0-or-later
"""APUMD command-line entry point.

This wrapper forces the TensorFlow backend and remaps ``apumd train`` to the
APUMD-specific two-stage training flow exposed as ``dp train-apumd``.
"""

from __future__ import annotations

import sys
import textwrap

from deepmd.main import (
    main as dp_main,
)


BACKEND_OPTIONS = {"-b", "--backend", "--pt", "--pd", "--jax"}
UNSUPPORTED_COMMANDS = {"freeze", "compress", "convert-from", "change-bias"}


def _print_help() -> None:
    help_text = textwrap.dedent(
        """\
        usage: apumd <command> [options]

        APUMD commands run on the TensorFlow backend.

        commands:
                    train          Run the full APUMD training flow
          test           Test a frozen APUMD/DeepMD model
          eval-desc      Evaluate descriptors
          model-devi     Calculate model deviation
          neighbor-stat  Calculate neighbor statistics
          show           Show model information

        examples:
                    apumd train input.json
          apumd test -m apumd_qnn/frozen_model.pb -s data

        notes:
                    apumd train runs the full CNN -> QNN pipeline via: dp --tf train-apumd
                    supported non-train commands map to: dp --tf <command>
                    unsupported for APUMD models: freeze, compress, convert-from, change-bias
        """
    )
    print(help_text)


def _print_train_help() -> None:
        help_text = textwrap.dedent(
                """\
                usage: apumd train [-h] [-v {DEBUG,3,INFO,2,WARNING,1,ERROR,0}] [-l LOG_PATH]
                                   [-i INIT_MODEL] [-r RESTART] [-f INIT_FRZ_MODEL]
                                   [--skip-neighbor-stat]
                                                     INPUT

                positional arguments:
                    INPUT                 the input parameter file in json format

                options:
                    -h, --help            show this help message and exit
                    -v {DEBUG,3,INFO,2,WARNING,1,ERROR,0}, --log-level {DEBUG,3,INFO,2,WARNING,1,ERROR,0}
                                                                set verbosity level by string or number, 0=ERROR,
                                                                1=WARNING, 2=INFO and 3=DEBUG
                    -l LOG_PATH, --log-path LOG_PATH
                                                                set log file to log messages to disk, if not
                                                                specified, the logs will only be output to console
                    -i INIT_MODEL, --init-model INIT_MODEL
                                                                Initialize the model by the provided path prefix of
                                                                checkpoint files.
                    -r RESTART, --restart RESTART
                                                                Restart the training from the provided prefix of
                                                                checkpoint files.
                    -f INIT_FRZ_MODEL, --init-frz-model INIT_FRZ_MODEL
                                                                Initialize the training from the frozen model.
                    --skip-neighbor-stat  Skip calculating neighbor statistics. Sel checking,
                                                                automatic sel, and model compression will be disabled.

                examples:
                    apumd train input.json
                    apumd train input.json --restart model.ckpt
                    apumd train input.json --init-model model.ckpt
                """
        )
        print(help_text)


def _validate_args(argv: list[str]) -> None:
    for index, arg in enumerate(argv):
        if arg not in BACKEND_OPTIONS:
            continue
        if arg in {"-b", "--backend"}:
            value = argv[index + 1] if index + 1 < len(argv) else None
            if value not in {None, "tensorflow", "tf"}:
                raise SystemExit("apumd only supports the TensorFlow backend")
        else:
            raise SystemExit("apumd only supports the TensorFlow backend")


def main(args: list[str] | None = None) -> None:
    argv = list(sys.argv[1:] if args is None else args)
    if not argv:
        _print_help()
        return
    if argv[0] in {"-h", "--help"}:
        _print_help()
        return

    _validate_args(argv)

    if argv[0] in UNSUPPORTED_COMMANDS:
        raise SystemExit(
            f"apumd does not support '{argv[0]}' for APUMD models"
        )

    if argv[0] == "train":
        if any(arg in {"-h", "--help"} for arg in argv[1:]):
            _print_train_help()
            return
        dp_main(["--tf", "train-apumd", *argv[1:]])
        return

    if argv[0] == "train-apumd":
        dp_main(["--tf", *argv])
        return

    dp_main(["--tf", *argv])


if __name__ == "__main__":
    main()