# OpenACC_Tutorial

## Run example

To build the project, follow these steps:

1. Open a terminal and navigate to the `openacc` directory.
2. Create a `build` directory by running the following command:
```bash
mkdir build
```
3. Navigate to the `build` directory:
```bash
cd build
```
4. Run the following command to configure the project:
```bash
cmake -DCMAKE_CXX_COMPILER=pgc++ ..
```
5. Build the project using the following command:
```bash
make -j4
```
6. Run executable
```bash
./openacc_tut
```

