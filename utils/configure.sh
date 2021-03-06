#!/bin/sh

# This file for Linux users, 
# launches CMake and creates configuration for
# Release and Debug modes.

echo
echo ============= Checking for CMake ============
echo

if (cmake --version); then
    echo "Found CMake"
    echo
else
    echo "Error: CMake not found, please install it (see http://www.cmake.org/)"
    exit 1
fi

# Parse command line arguments

cmake_options=
build_name_suffix=
while [ -n "$1" ]; do
    case "$1" in
        --with-*=*)
            cmake_option=`echo "$1" | sed 's/--with-\([^=]*\)=\(.*\)$/-DVORPALINE_WITH_\U\1\E:STRING="\2"/'`
            cmake_options="$cmake_options $cmake_option"
            shift
            ;;
        --with-*)
            cmake_option=`echo "$1"  | awk '/a/ {sub(/--with-/, ""); print "-DVORPALINE_WITH_"toupper($1)":BOOL=TRUE"'}`
            cmake_options="$cmake_options $cmake_option"
            shift
            ;;
        --help-platforms)
            echo "Supported platforms:"
            for i in `find geogram/cmake/platforms/* -type d`
            do
                if [ $i != "xxxgeogram/cmake/platforms" ]
                then
                    echo "*" `basename $i`
                fi
            done
            exit
            ;;
        --build_name_suffix=*)
            build_name_suffix=`echo "$1" | sed 's/--build_name_suffix=\(.*\)$/\1/'`
            shift
            ;; 
            
        --help)
            cat <<END
NAME
    configure.sh

SYNOPSIS
    Prepares the build environment for Hexdom/Vorpaline.
    
    - For Unix builds, the script creates 2 build trees for Debug and Release
    build in a 'build' sub directory under the project root.

USAGE
    configure.sh [options] [build-platform]

OPTIONS

    --help
        Prints this page.

    --build_name_suffix=suffix-dir
        Add a suffix to define the build directory

EXAMPLES
    ./configure.sh
    ./configure.sh --with-asan

PLATFORM
    Build platforms supported by Hexdom: use configure.sh --help-platforms
END
            exit
            ;;
            
        -*)
            echo "Error: unrecognized option: $1"
            return 1
            ;;
        *)
            break;
            ;;
    esac
done

# Check the current OS
os="$1"
if [ -z "$os" ]; then
    os=`uname -a`
    case "$os" in
        Linux*x86_64*)
            os=Linux64-gcc-dynamic
            ;;
        Linux*amd64*)
            os=Linux64-gcc-dynamic
            ;;
        Linux*i586*|Linux*i686*)
            os=Linux32-gcc-dynamic
            ;;
        Darwin*)
            os=Darwin-clang-dynamic
            ;;
        *)
            echo "Error: OS not supported: $os"
            exit 1
            ;;
    esac
fi

# Generate the Makefiles
for config in Release RelWithDebInfo Debug; do
   platform=$os-$config
   echo
   echo ============= Creating makefiles for $platform ============
   echo

   build_dir=build/$platform$build_name_suffix
   mkdir -p $build_dir
   (cd $build_dir;
    cmake \
        -DCMAKE_BUILD_TYPE:STRING=$config \
        -DVORPALINE_PLATFORM:STRING=$os \
    $cmake_options ../../)
done

echo
echo ============== Hexdom build configured ==================
echo

cat << EOF
To build hexdom:
  - go to build/$os-Release$build_name_suffix or build/$os-Debug$build_name_suffix
  - run 'make' or 'cmake --build .'

Note: local configuration can be specified in CMakeOptions.txt
(see CMakeOptions.txt.sample for an example)
You'll need to re-run configure.sh if you create or modify CMakeOptions.txt

EOF

