#!/bin/bash
############################################################
#                                                          #
# Intel FPGA SoC Embedded Command Shell                    #
#                                                          #
#                                                          #
# Copyright (c) 2018 Intel Corporation                     #
# All Rights Reserved.                                     #
#                                                          #
############################################################


############################################################
#
# Get the Root SOCEDS directory
#

if [ -n "${COMSPEC}" ]; then
    _IS_WINDOWS=1
 
    if [ "${SOCEDS_DESTROY_PATH}" == "1" ]; then	
        export ORIGINAL_PATH="${PATH}"
        PATH="/bin:/usr/bin"
    fi
fi

if [ "${_IS_WINDOWS}" = "1" ] && [ -n "$(which cygpath 2>/dev/null)" ]; then
    _IS_CYGWIN=1
fi

#_SOCEDS_ROOT=$(cd "$(dirname "${0}")" && echo "$(pwd 2>/dev/null)")
_SOCEDS_ROOT=${INTELFPGAOCLSDKROOT}/../embedded

if [ ! -d "${_SOCEDS_ROOT}" ]; then
    echo "${_SOCEDS_ROOT} not found. Invalid or corrupt SOCEDS Install" 1>&2
    exit 1
fi

export SOCEDS_DEST_ROOT="${_SOCEDS_ROOT}"
if [ "${_IS_CYGWIN}" == "1" ]; then
    SOCEDS_DEST_ROOT="$(cygpath -m "${SOCEDS_DEST_ROOT}" 2>/dev/null)"
fi

source "${_SOCEDS_ROOT}/env.sh"

############################################################


unset _SOCEDS_ROOT
unset _IS_WINDOWS
unset _IS_CYGWIN


if [ -n "$*" ]; then
    exec "$@"
else
    # Use bash --norc to get a clean shell
    # Use bash --rcfile <bashrc> to for a user rcfile
    # Default to using ~/.bashrc
    cd fpga_build
    make
    cd ../
fi

