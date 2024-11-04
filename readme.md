## Build
```
cmake . -B build -Dbout++_DIR=/Users/power8/Documents/01_code/04_bout/build-bout
```
then 
```
cmake --build build
```

## Run
```
mpirun -np 2 ./churn -d ../data
```
