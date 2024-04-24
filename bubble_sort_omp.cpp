#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

void bubbleSort(int arr[], int n) {
    for (int i = 0; i < n - 1; i++) {
        for (int j = 0; j < n - i - 1; j++) {
            if (arr[j] > arr[j + 1]) {
                int temp = arr[j];
                arr[j] = arr[j + 1];
                arr[j + 1] = temp;
            }
        }
    }
}

void parallelBubbleSort(int arr[], int n) {
    int i, j;
    for (i = 0; i < n - 1; i++) {
#pragma omp parallel for shared(arr) private(j)
        for (j = 0; j < n - i - 1; j++) {
            if (arr[j] > arr[j + 1]) {
                int temp = arr[j];
                arr[j] = arr[j + 1];
                arr[j + 1] = temp;
            }
        }
    }
}

int main() {
    int n = 10000;
    int arr[10000], arr_copy[10000];

    for (int i = 0; i < n; i++) {
        arr[i] = rand() % 10000;
    }

    for (int i = 0; i < n; i++) {
        arr_copy[i] = arr[i];
    }

    double start_time = omp_get_wtime();
    parallelBubbleSort(arr, n);
    double parallel_bubble_sort_time = omp_get_wtime() - start_time;

    start_time = omp_get_wtime();
    bubbleSort(arr_copy, n);
    double sequential_bubble_sort_time = omp_get_wtime() - start_time;

    // Output the times
    printf("Sequential Bubble Sort Time: %f seconds\n", sequential_bubble_sort_time);
    printf("Parallel Bubble Sort Time: %f seconds\n", parallel_bubble_sort_time);

    return 0;
}
