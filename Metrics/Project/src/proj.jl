using CSV,DataFrames;

DataFolder = "/media/jintu/MyStuff/Work/Datasets/ECON8206_Project/";
raw_asm = CSV.read(string(DataFolder,"ASM_Millons.csv"),null="\-");
asm1 = readtable(string(DataFolder,"ASM_Millons.csv"))
