#!/bin/bash

function usage {
	echo "usage: $0 [-od] [-r <string>] [-p <string>]"
	echo " -o        Overwrite previous file names"
	echo " -d        Dry run, no GaussianObject code executed"
        echo " -D        Dry run with run_name project directory created"
	echo " -r        Set run_name, i.e. the iteration of project (default: test)"
        echo " -p        Select project (default: frame_42sec)"
        echo " -s        Sparsity (default: 4)"
        echo " -e        Use Dust3r"
        echo " -E        Use extrinsics from another dust3r run (skips pred_poses step)"
        echo " -z        Resolution to use after downsizing (default 1 i.e. original)"
        echo " -P        Print all commands back to back for immediate use"
	echo " -X        Stop at this step of the process"
        echo " -g        Indicates execution in VSCode and using terminal debugger"
        exit
}

run_name="test"
project="frame_42sec"
sparsity=4
resolution=1

overwrite=0
dry=0
dust=0
stopat=0
vsdebugger=0
while getopts ":odr:p:hs:DeEz:PX:g" o
do
        case "$o" in
                o)
                        echo "Enabling overwriting..."
			                  overwrite=1
                        ;;
                d)
                        echo "Dry run..."
			                  dry=1
                        ;;
                D)
                        echo "Dry run with project directory setup..."
                        dry=2
                        ;;
		            r)
                        run_name="$OPTARG"
                        echo "run_name name set to $run_name"
                        ;;
                p)
                        project="$OPTARG"
                        echo "project set to $project"
                        ;;
                s)
                        sparsity="$OPTARG"
                        echo "sparsity set to $sparsity"
                        ;;
                h)
                        usage
                        ;;
                e)
                        dust=1
                        echo "using dust3r"
                        ;;
                E)
                        dust=2
                        echo "using extrinsics copied from another dust3r run"
                        ;;
                z)
                        resolution="$OPTARG"
                        echo "resolution set to $resolution"
                        ;;
                P)
                        print_raw=1
                        echo "Printing raw commands"
                        ;;
		X)
			stopat="$OPTARG"
			echo "stopping when we hit $stopat"
			;;
                g)
                        echo "VSCode debugging enabled"
                        vsdebugger=1
                        ;;

                :)
                        echo "ERROR: Option -$OPTARG requires an argument"
                        usage
                        ;;
                \?)
                        echo "Invalid option -$OPTARG"
                        usage
                        ;;
        esac
done

dir_name="$run_name"-"$project"
echo "New project name: $dir_name"
proceed=1
base_dir="/home/leelab/Desktop/GaussianObject"
data_dir="$base_dir/data/natedata/$dir_name"

progs=("downsample" "preprocessing" "visual" "first_train"\
        "first_render" "first_render_cont" "loo1"\
        "loo2" "lora" "repair_train" "final_render"\
        "final_render_cont")

if [[ $dust == 1 ]]
then
        progs=("downsample" "pred_poses" "first_train"\
        "first_render" "first_render_cont" "loo1"\
        "loo2" "lora" "repair_train" "final_render"\
        "final_render_cont")
fi

if [[ $dust == 2 ]]
then
        progs=("downsample" "first_train"\
        "first_render" "first_render_cont" "loo1"\
        "loo2" "lora" "repair_train" "final_render"\
        "final_render_cont")
fi

if [[ $print_raw == 1 ]]
then
        if [[ $dry == 2 ]];
        then
                echo "By choosing this combination of -D and -P, you will exit before the directory creation step."
        fi
	colors=($'\033[0;31m'  # Red
        	$'\033[0;32m'  # Green
  		$'\033[0;33m'  # Yellow
  		$'\033[0;34m'  # Blue
  		$'\033[0;35m'  # Magenta
  		$'\033[0;36m'  # Cyan
  		$'\033[0;37m'  # White
	)
	NC=$'\033[0m'
        cmd_array=()
        for i in ${!progs[@]}
        do
            if [[ "${progs[$i]}" == "$stopat" ]]
	    then
		    echo "Asked to stop here..."
		    break
	    fi

	    color="${colors[$((i % ${#colors[@]}))]}"
	    command="$(python nate_scripts/strPrinter.py ${progs[$i]} $dir_name $data_dir $sparsity $dust $resolution $vsdebugger)"
	    cmd_array+=("${color}$command${NC};")
        done
	printf "%s " "${cmd_array[@]}"
        exit
fi



if [[ $dry == 1 ]]
then
        python nate_scripts/strPrinter.py "all" $dir_name "$base_dir/data/natedata/$dir_name" $sparsity $dust $resolution $vsdebugger
        exit
fi

echo "Checking for previously existing runs with the same name..."

if [[ -d "$base_dir"/data/natedata/"$dir_name" ]] 
then
	echo "$dir_name already found in data/natedata"
	proceed=0
	if [[ $overwrite == 1 ]]
	then
		rm -r "$base_dir"/data/natedata/"$dir_name"
		proceed=1
	fi
fi

if [[ -d "$base_dir"/output/controlnet_finetune/"$dir_name" ]] 
then
	echo "$dir_name already found in output/controlnet_finetune"
	proceed=0
	if [[ $overwrite == 1 ]]
	then
		rm -r "$base_dir"/output/controlnet_finetune/"$dir_name"
		proceed=1
	fi
fi

if [[ -d "$base_dir"/output/gaussian_object/"$dir_name" ]] 
then
        echo "$dir_name already found in output/gaussian_object"
	proceed=0
	if [[ $overwrite == 1 ]]
	then
		rm -r "$base_dir"/output/gaussian_object/"$dir_name"
		proceed=1
	fi
fi

if [[ -d "$base_dir"/output/gs_init/"$dir_name" ]] 
then
        echo "$dir_name already found in output/gs_init"
	proceed=0
	if [[ $overwrite == 1 ]]
	then
		rm -r "$base_dir"/output/gs_init/"$dir_name"
		proceed=1
	fi
fi

if [[ $proceed == 1 ]]
then
        echo "Overwrite check completed"
else
        echo "Please fix naming problems to continue"
        exit
fi

echo "Creating a copy of $project at $data_dir"

cp -i -r "$base_dir/data/natedata/$project" "$data_dir"
find "$data_dir" -name ".DS_Store" -print -delete
echo ""

# Storing run times for each program in this array
declare -A durations

echo -e "\nMain functions starting at "
date +%T


for prog in ${progs[@]}
do
        echo -e "\n$prog"
	if [[ "$stopat" == "$prog" ]]
	then
		echo "Asked to stop early at this step..."
		break
	fi
        command="$(python nate_scripts/strPrinter.py $prog $dir_name $data_dir $sparsity $dust $resolution $vsdebugger)"
        if [[ $dry -eq 0 ]]
        then
            start=$(date +%s.%N)
            time eval "$command"
            end=$(date +%s.%N)
            duration=$(echo "($end - $start)" | bc)
            durations[$prog]=$duration
        else
            echo $command
        fi
        echo "-----------------------------"
done

echo "Main functions ending at"
date +%T

if [[ $dry -eq 0 ]]
then
    echo "===== Summary of Runtimes ====="
    for prog in "${progs[@]}"; do
        echo "$prog: ${durations[$prog]} seconds"
    done
fi

exit


