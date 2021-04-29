#!/bin/bash

MULT="100"

STABLE=19
NIGHTLIES=20
LOWEST=18

CTK_VERSION=11.0

RAPIDS_VERSION="0.$STABLE"
RAPIDS_RESULT=$STABLE
 
echo "PLEASE READ"
echo "********************************************************************************************************"
echo "Changes:"
echo "1. IMPORTANT SCRIPT CHANGES: Colab has updated to Python 3.7, and now runs our STABLE and NIGHTLY versions (0.$STABLE and 0.$NIGHTLIES)!  PLEASE update your older install script code as follows:"
echo "	!bash rapidsai-csp-utils/colab/rapids-colab.sh 0.19"
echo ""
echo "	import sys, os"
echo ""
echo "	dist_package_index = sys.path.index('/usr/local/lib/python3.7/dist-packages')"
echo "	sys.path = sys.path[:dist_package_index] + ['/usr/local/lib/python3.7/site-packages'] + sys.path[dist_package_index:]"
echo "	sys.path"
echo "	exec(open('rapidsai-csp-utils/colab/update_modules.py').read(), globals())"
echo ""
echo "2. IMPORTANT NOTICE: CuGraph's Louvain requires a Volta+ GPU (T4, V100).  If you get a P4 or P100 and intend to use Louvain, please FACTORY RESET your instance and try to get a compatible GPU"
echo "3. Default stable version is now 0.$STABLE.  Nightly is now 0.$NIGHTLIES."
echo "3. You can declare your RAPIDSAI version as a CLI option and skip the user prompts (ex: '0.$STABLE' or '0.$NIGHTLIES', between 0.$LOWEST to 0.$NIGHTLIES, without the quotes): "
echo '        "!bash rapidsai-csp-utils/colab/rapids-colab.sh <version/label>"'
echo "        Examples: '!bash rapidsai-csp-utils/colab/rapids-colab.sh 0.$STABLE', or '!bash rapidsai-csp-utils/colab/rapids-colab.sh stable', or '!bash rapidsai-csp-utils/colab/rapids-colab.sh s'"
echo "                  '!bash rapidsai-csp-utils/colab/rapids-colab.sh 0.$NIGHTLIES, or '!bash rapidsai-csp-utils/colab/rapids-colab.sh nightly', or '!bash rapidsai-csp-utils/colab/rapids-colab.sh n'"
echo "Enjoy using RAPIDS!  If you have any issues with or suggestions for RAPIDSAI on Colab, please create a bug request on https://github.com/rapidsai/rapidsai-csp-utils/issues/new."
echo "For a near instant entry into a RAPIDS Library experience, or if we haven't fixed a fatal issue yet, please use https://app.blazingsql.com/.  Thanks!"


install_RAPIDS () {
    echo "Checking for GPU type:"
    python cusignal-icassp-tutorial/colab/env-check.py
    if [ $? -ne 0 ]; then
      echo -e "\nError during environment check! Exiting"
      exit
    fi

    if [ ! -f Miniconda3-4.5.4-Linux-x86_64.sh ]; then
       
        # intall miniconda
        echo "Installing conda"
        wget -nc https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
        chmod +x Miniconda3-latest-Linux-x86_64.sh
        bash ./Miniconda3-latest-Linux-x86_64.sh -b -f -p /usr/local

        # pin python3.7
        echo "python 3.7.*" > /usr/local/conda-meta/pinned
		
		
		# Install RAPIDSAI Libraries
        if (( $RAPIDS_RESULT == $NIGHTLIES )) ;then  # Newest nightly packages.  UPDATE EACH RELEASE!
        echo "Installing RAPIDS $RAPIDS_VERSION packages from the nightly release channel"
        echo "Please standby, this will take a few minutes..."
        # install RAPIDS packages
            conda install -y --prefix /usr/local \
                -c rapidsai-nightly -c nvidia -c conda-forge -c defaults \
                python=3.7 cudatoolkit=$CTK_VERSION \
                cusignal=$RAPIDS_VERSION \
                llvmlite gcsfs openssl
        elif (( $RAPIDS_RESULT == $LOWEST )) ; then
            echo "Installing RAPIDS $RAPIDS_VERSION packages from the stable release channel"
            echo "Please standby, this will take a few minutes..."
            # install RAPIDS packages
            conda install -y --prefix /usr/local \
                -c rapidsai -c nvidia -c conda-forge -c defaults \
                python=3.7 cudatoolkit=$CTK_VERSION \
                cusignal=$RAPIDS_VERSION \
                llvmlite gcsfs openssl
        else  # Stable packages 0.18 uses xgboost 1.3.3
            echo "Installing RAPIDS $RAPIDS_VERSION packages from the stable release channel"
            echo "Please standby, this will take a few minutes..."
            # install RAPIDS packages
            conda install -y --prefix /usr/local \
                -c rapidsai/label/main -c rapidsai -c nvidia -c conda-forge -c defaults \
                python=3.7 cudatoolkit=$CTK_VERSION \
                cusignal=$RAPIDS_VERSION \
                llvmlite gcsfs openssl
        fi
          
        echo "Copying shared object files to /usr/lib"
        # copy .so files to /usr/lib, where Colab's Python looks for libs
        python cusignal-icassp-tutorial/colab/copy_libs.py
    fi

    echo ""
    echo "************************************************"
    echo "Your Google Colab instance has RAPIDS installed!"
    echo "************************************************"
}

rapids_version_check () {
    
  if  [ $RESPONSE == "NIGHTLY" ]|| [ $RESPONSE == "nightly" ]  || [ $RESPONSE == "N" ] || [ $RESPONSE == "n" ] ; then
    RAPIDS_VERSION="0.$NIGHTLIES"
    RAPIDS_RESULT=$NIGHTLIES
    echo "Starting to prep Colab for install RAPIDS Version $RAPIDS_VERSION nightly"
  elif [ $RESPONSE == "STABLE" ]|| [ $RESPONSE == "stable" ]  || [ $RESPONSE == "S" ] || [ $RESPONSE == "s" ] ; then
    RAPIDS_VERSION="0.$STABLE"
    RAPIDS_RESULT=$STABLE
    echo "Starting to prep Colab for install RAPIDS Version $RAPIDS_VERSION stable"
  else
    RAPIDS_RESULT=$(awk '{print $1*$2}' <<<"${RESPONSE} ${MULT}")
    if (( $RAPIDS_RESULT > $NIGHTLIES )) ; then
      RAPIDS_VERSION="0.$NIGHTLIES"
      RAPIDS_RESULT=$NIGHTLIES
      echo "RAPIDS Version modified to $RAPIDS_VERSION nightly"
    elif (($RAPIDS_RESULT < $LOWEST)) ; then
      RAPIDS_VERSION="0.$LOWEST"
      RAPIDS_RESULT=$LOWEST
      echo "RAPIDS Version modified to $RAPIDS_VERSION stable"
    elif (($RAPIDS_RESULT >= $LOWEST)) &&  (( $RAPIDS_RESULT <= $NIGHTLIES )) ; then
      RAPIDS_VERSION="0.$RAPIDS_RESULT"
      echo "RAPIDS Version to install is $RAPIDS_VERSION"
    else
      echo "You've entered and incorrect RAPIDS version.  please make the necessary changes and try again"
    fi
  fi
}

if [ -n "$1" ] ; then
  RESPONSE=$1
  rapids_version_check
  install_RAPIDS
else
  echo "As you didn't specify a RAPIDS version, please enter in the box your desired RAPIDS version (ex: '0.$STABLE' or '0.$NIGHTLIES', between 0.$LOWEST to 0.$NIGHTLIES, without the quotes)"
  echo "and hit Enter. If you need stability, use 0.$STABLE. If you want bleeding edge, use our nightly version (0.$NIGHTLIES), but understand that caveats that come with nightly versions."
  read RESPONSE
  rapids_version_check
  install_RAPIDS
fi
