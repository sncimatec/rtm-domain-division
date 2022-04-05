# RTM Domain Division in a multi-GPU environment

The folders of RTM codes has four subdirectories: library (lib), velocity models (build), include and source (src).

## COMPILE

To compile and run the RTM code needs fist compile. To compile go to src folder, and run make allclean, so run make. You need to be inside the folder that contains the file. If the compilation has finished without errors will are produced two executable files on build folder mod_main and rtm_main  in build folder.

## RUN
Before run rtm_main, you need to run modeling step: 
```
$ ./mod_main ./models/<choose a model>/input.dat
```

After you can run normally the migration step:
```
$ ./rtm_main ./models/<choose a model>/input.dat
```
