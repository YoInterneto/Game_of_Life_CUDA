
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "cuda.h"

#include <time.h>
#include <stdio.h>
#include <iostream>
#include <stdlib.h>

#ifndef __CUDACC__
#define __CUDACC__
#endif


__global__ void movimientoCelular(char* matriz, char* matrizResultado, int fila, int columna);

void imprimirMatriz(char* matriz, int dimension, int columna);

void rellenarMatriz(char* matriz, int dimension);

int main(int arg, char* argv[])
{
 
    //Comprueba que haya solo el numero de argumento permitidos
    if (arg != 4){
        printf("\nERROR: El numero de argumentos es erróneo (.exe <-a/-m> <fila> <columna>)\n");
    }
    else {

        //Conversion de argumentos a int
        char* filaPuntero = argv[2];
        int fila = atoi(filaPuntero);
        char* columnaPuntero = argv[3];
        int columna = atoi(columnaPuntero);

        //Dimension de la matriz
        int dimension = columna * fila;

        //Matrices
        char* matriz = NULL;
        char* matrizResultado = NULL;

        matriz = (char*)malloc(sizeof(char) * dimension);
        matrizResultado = (char*)malloc(sizeof(char) * dimension);

        //Punteros matriz
        char* matriz_d;
        char* matrizResultado_d;

        //Dimensiones del bloque
        dim3 blockDim(fila,columna); //Para ver arriba o abajo tendremos que sumarle el valor de la columna-1 o columna+1
                                     //Matriz(23x24)
                                     //ThreadIdx.x sera la fila (23)
                                     //ThreadIdx.y sera la columna (24)

        //Booleano para saber si el usuario quiere manual o automatico, por defecto automatico
        bool manual = false;

        //Comprueba que los numeros de columna y fila son correctos
        if (columna <= 0 | fila <= 0) {
            printf("\nERROR: La fila/columna tiene que ser un entero positivo.\n");
        }
        //Comprueba que se haya introducido el parametro de ejecucion correcto 
        else if ((strcmp("-m", argv[1]) & strcmp("-a", argv[1])) != 0) {
            printf("\nERROR: Argumentos validos solo -m[manual] o -a[automatico]\n");
        }
        //Una vez comprobado todo empezamos con la ejecucion
        else {

            printf("\n[Matriz(%dx%d) Dimension(%d)] [modo: %s] \n", fila, columna, dimension, argv[1]);

            if (strcmp("-m", argv[1]) == 0) {
                manual = true;
            }

            //Rellenamos el tablero con celulas muertas y vivas
            rellenarMatriz(matriz,dimension);
            printf("\n***TABLERO INICIAL***\n");
            imprimirMatriz(matriz, dimension, columna);
            imprimirMatriz(matrizResultado, dimension, columna);

            //Iniciamos el kernel
            cudaMalloc(&matriz_d, sizeof(matriz));
            cudaMalloc(&matrizResultado_d, sizeof(matrizResultado));
            cudaMemcpy(matriz_d, &matriz, sizeof(matriz), cudaMemcpyHostToDevice);
            cudaMemcpy(matrizResultado_d, &matrizResultado, sizeof(matrizResultado), cudaMemcpyHostToDevice);

            int generaciones = 1; //Cuenta cuantas iteraciones (generaciones) han habido

            //Se podria poner el resultado de una funcion que cambiara el valor de un bool terminado que dijera cuando no quedan
            //mas celulas vivas por ejemplo bool terminado = false ... while(!terminado) ... if(terminar(matriz)) terminado = true
            while (generaciones < 2) {

                if (generaciones == 1) {
                    movimientoCelular << <1, blockDim >> > (matriz_d, matrizResultado_d, fila, columna);
                }
                else {
                    movimientoCelular << <1, blockDim >> > (matrizResultado_d, matrizResultado_d, fila, columna);
                }

                cudaMemcpy(&matrizResultado, matrizResultado_d, sizeof(matrizResultado), cudaMemcpyDeviceToHost);

                printf("\nGeneracion: %d\n", generaciones);
                imprimirMatriz(matrizResultado, dimension, columna);

                //Si el usuario marca como manual, cada generacion tendra que pulsar alguna tecla para continuar
                if (manual) {
                    system("pause");
                }

                generaciones ++;
            }

            cudaFree(matriz_d);
            cudaFree(matrizResultado_d);
        }

        //Liberamos los arrays
        free(matriz);
        free(matrizResultado);

    }
}

__global__ void movimientoCelular(char* matriz, char* matrizResultado, int fila, int columna) {

    int posicion = threadIdx.x * columna + threadIdx.y;

    char* celula = matriz + posicion;
    char* celulaCambio = matrizResultado + posicion;

    matrizResultado[posicion] = 'X';
}

void imprimirMatriz(char* matriz, int dimension, int columna) {
    
    for (int i = 0; i < dimension; i ++) {

        printf(" %c ",*(matriz+i));

        if ((i + 1) % columna == 0) {
            printf("\n");
        }
    }
}

void rellenarMatriz(char* matriz, int dimension) {
    
    srand(time(0));

    for (int i = 0; i < dimension; i++) {

        char* celula = matriz + i;

        int random = rand() % dimension + 1;

        if (random % 8 == 0 | random % 9 == 0) {
            *celula = 'X';
        }
        else {
            *celula = 'O';
        }

    }
}


