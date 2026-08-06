set script_dir [file dirname [file normalize [info script]]]
set project_root [file normalize [file join $script_dir ".."]]

if {![info exists ::env(GEMINI_FPGA_HLS_ROOT)] ||
    $::env(GEMINI_FPGA_HLS_ROOT) eq ""} {
    error "GEMINI_FPGA_HLS_ROOT is required"
}
set fpga_root [file normalize $::env(GEMINI_FPGA_HLS_ROOT)]
set hls_dir [file join $fpga_root "hls"]
set vision_root [file normalize [file join $fpga_root ".." ".." ".." ".."]]
set xf_include [file join $vision_root "vision" "L1" "include"]
set accelerator [file join $hls_dir "gemini335_sgbm_accel.cpp"]
set config [file join $hls_dir "gemini335_sgbm_config.hpp"]
set sgbm_header [file join $xf_include "imgproc" "xf_sgbm.hpp"]

foreach required [list $accelerator $config $sgbm_header] {
    if {![file exists $required]} {
        error "Missing required FPGA/HLS source: $required"
    }
}

set build_root [file join $project_root "build"]
file mkdir $build_root
cd $build_root
open_project -reset gemini335_dataset_hls_csim
set_top gemini335_sgbm_accel
set cflags "-O3 -I$hls_dir -I$xf_include -std=c++0x"
add_files $accelerator -cflags $cflags
add_files -tb [file join $script_dir "tb_gemini335_dataset.cpp"] -cflags $cflags

open_solution -reset sol1
set_part "xczu5ev-sfvc784-2-i"
create_clock -period 6.5 -name default
csim_design -clean
exit
