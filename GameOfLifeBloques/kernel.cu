
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "cuda.h"
#include "./common/book.h"

#include <time.h>
#include <stdio.h>
#include <iostream>
#include <stdlib.h>

void imprimirMatriz(char** matriz, int fila, int columna);

void rellenarMatriz(char** matriz, int fila, int columna);

int contarVivas(char** matriz, int fila, int columna);


int main(int arg, char* argv[])
{

    //Comprueba que haya solo el numero de argumento permitidos
    if (arg != 4) {
        printf("\nERROR: El numero de argumentos es erroneo (.exe <-a/-m> <fila> <columna>)\n");
    }
    else {

        //Conversion de argumentos a int
        char* filaPuntero = argv[2];
        int fila = atoi(filaPuntero);
        char* columnaPuntero = argv[3];
        int columna = atoi(columnaPuntero);

        //Inicializamos cudaDeviceProp para coger las propiedades de la tarjeta
        cudaDeviceProp propiedades;
        HANDLE_ERROR(cudaGetDeviceProperties(&propiedades, 0));

        //Dimension de la matriz
        int dimension = columna * fila;

        //Matrices
        char** matriz = NULL;
        char** matrizResultado = NULL;

        matriz = (char**) malloc(sizeof(char) * dimension);
        matrizResultado = (char**) malloc(sizeof(char) * dimension);

        //Creamos las filas de las matrices
        for (int i = 0; i < columna; i++) {
            matriz[i] = (char*) malloc(sizeof(char) * columna);
            matrizResultado[i] = (char*) malloc(sizeof(char) * columna);
        }

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
            rellenarMatriz(matriz, fila, columna);

            printf("\n***TABLERO INICIAL***\n");
            printf("Celulas vivas: %d\n", contarVivas(matriz, fila, columna));
            imprimirMatriz(matriz, fila, columna);
        }

        //Liberamos los arrays
        free(matriz);
        free(matrizResultado);

    }
}

void imprimirMatriz(char** matriz, int fila, int columna) {

    for (int i = 0; i < fila; i++) {
        for (int j = 0; j < columna; j++) {
            printf(" %c ",matriz[i][j]);
        }

        printf("\n");
    }
}

void rellenarMatriz(char** matriz, int fila, int columna) {

    srand(time(0));

    for (int i = 0; i < fila; i++) {
        for (int j = 0; j < columna; j++) {

            int random = rand() % (fila*columna) + 1;

            if (random % 3 == 0 && random % 2 == 0) {

                matriz[i][j] = 'X';
            }
            else {
                matriz[i][j] = 'O';
            }
        }
    }
}

int contarVivas(char** matriz, int fila, int columna) {

    int contador = 0;

    for (int i = 0; i < fila; i++) {
        for (int j = 0; j < columna; j++) {
            if (matriz[i][j] == 'X') {
                contador++;
            }
        }
    }

    return contador;
}