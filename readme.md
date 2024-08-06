## Build
```
cmake . -B build -Dbout++_DIR=/path/to/bout/build/directory
```
then 
```
cmake --build build
```

## Run
```
mpirun -np 2 ./churn -d ../data
```
