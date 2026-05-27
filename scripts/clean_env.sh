#!/usr/bin/env bash
# Source this (or run via `env -i bash --noprofile -c ...`) to strip every
# environment knob conda's base activation leaves behind that MadGraph or
# LHAPDF will otherwise bake into their build artefacts. Symptoms of NOT
# doing this: gfortran link errors against -lLHAPDF (undefined `std::*`),
# RPATH entries pointing at `/home/vince/miniforge3/lib`, and Fortran
# binaries that won't run on a fresh user machine.
#
# Usage:
#   . scripts/clean_env.sh   # in-shell scrub
# or:
#   scripts/clean_env.sh -- some_command arg1 arg2
#
# The script does not touch your shell after returning unless sourced.

# Variables conda sets on activation that leak into compilers/cmake/etc.
_CONDA_LEAKED_VARS=(
    CONDA_EXE CONDA_PREFIX CONDA_DEFAULT_ENV CONDA_PROMPT_MODIFIER
    CONDA_PYTHON_EXE CONDA_SHLVL CONDA_TOOLCHAIN_BUILD CONDA_TOOLCHAIN_HOST
    CMAKE_ARGS CMAKE_PREFIX_PATH
    CC CXX CPP FC F77 F90 LD AR AS NM RANLIB STRIP
    GCC GXX GFORTRAN
    GCC_AR GCC_NM GCC_RANLIB GPROF
    CXXFLAGS CFLAGS CPPFLAGS LDFLAGS FFLAGS FCFLAGS
    CPATH C_INCLUDE_PATH CPLUS_INCLUDE_PATH LIBRARY_PATH LD_LIBRARY_PATH
    PKG_CONFIG_PATH XML_CATALOG_FILES
    ADDR2LINE BUILD HOST PYTHIA8DATA
    build_alias host_alias
    DEBUG_CFLAGS DEBUG_CXXFLAGS DEBUG_FFLAGS DEBUG_FORTRANFLAGS DEBUG_CPPFLAGS
    SIZE STRINGS OBJDUMP OBJCOPY READELF ELFEDIT GCC_TOOLCHAIN_PREFIX
    CONDA_BUILD_SYSROOT MESON_ARGS HOST_NAME
    _CONDA_EXE _CONDA_ROOT MAMBA_EXE MAMBA_ROOT_PREFIX
    PYTHIA8 PYTHIA8DATA GSETTINGS_SCHEMA_DIR GSETTINGS_SCHEMA_DIR_CONDA_BACKUP
    NVCC_PREPEND_FLAGS LDFLAGS_LD
    CC_FOR_BUILD CXX_FOR_BUILD CPP_FOR_BUILD F77_FOR_BUILD FC_FOR_BUILD
    AS_FOR_BUILD AR_FOR_BUILD LD_FOR_BUILD NM_FOR_BUILD STRIP_FOR_BUILD
    RANLIB_FOR_BUILD READELF_FOR_BUILD OBJCOPY_FOR_BUILD OBJDUMP_FOR_BUILD
    SIZE_FOR_BUILD STRINGS_FOR_BUILD ADDR2LINE_FOR_BUILD ELFEDIT_FOR_BUILD
    GPROF_FOR_BUILD HOST_FOR_BUILD GCC_FOR_BUILD GXX_FOR_BUILD GFORTRAN_FOR_BUILD
    PKG_CONFIG PKG_CONFIG_PATH_NO_BUILD_SYSROOT
    ROOTSYS PYTHIA8_DIR PYTHIADATA CXXFILT
)

# Last-resort: drop anything still pointing at miniforge/anaconda. Never
# touch PATH itself (the prior block already cleaned it, and unsetting PATH
# here would break the tr/grep below for any caller).
while IFS='=' read -r _name _val; do
    case "$_name" in
        PATH|HOME|USER|SHELL|TERM) continue ;;
    esac
    case "$_val" in
        *miniforge*|*/conda/*|*x86_64-conda-linux-gnu*)
            unset "$_name"
            ;;
    esac
done < <(env)

for _v in "${_CONDA_LEAKED_VARS[@]}"; do unset "$_v"; done

# Strip miniforge / anaconda from PATH FIRST, then prepend system bins so the
# follow-up scrub below has working tr/grep/sed.
PATH="$(echo "$PATH" | tr ':' '\n' | grep -vE '(miniforge|anaconda|/conda/)' | tr '\n' ':' | sed 's/:$//')"
export PATH="/usr/bin:/bin:$PATH"

unset _CONDA_LEAKED_VARS _v

# If invoked as a script with arguments after `--`, exec them in this clean env.
if [ "${1:-}" = "--" ]; then
    shift
    exec "$@"
fi
