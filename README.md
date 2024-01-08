# DD2360: Team Project - Project Group 3

## Compile and Run the Code

### For original.cu or original_unified.cu files:

1. Navigate to the `src` folder:

    ```
    cd src
    ```

2. Compile the code:

    ```
    nvcc -o original_unified original_unified.cu
    ```

3. Run the compiled code with a specific input file (e.g., matrix100.txt):

    ```
    ./original_unified -f ../data/matrix100.txt
    ```

### For cusolve.cu or cusolve_unified.cu files:

1. Navigate to the `src` folder:

    ```
    cd src
    ```

2. Compile the code with the cusolver library:

    ```
    nvcc -o cusolve_unified cusolve_unified.cu -lcusolver
    ```

3. Run the compiled code with a specific input file (e.g., matrix100.txt):

    ```
    ./cusolve_unified ../data/matrix100.txt
    ```
