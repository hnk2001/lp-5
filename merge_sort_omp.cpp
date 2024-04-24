#include <stdio.h>
#include <vector>
#include <stdlib.h>
#include <omp.h>

void merge(std::vector<int>& arr, int l, int m, int r) {
    int n1 = m - l + 1;
    int n2 = r - m;

    std::vector<int> L(n1), R(n2);

    for (int i = 0; i < n1; i++)
        L[i] = arr[l + i];
    for (int j = 0; j < n2; j++)
        R[j] = arr[m + 1 + j];

    int i = 0, j = 0, k = l;
    while (i < n1 && j < n2) {
        if (L[i] <= R[j]) {
            arr[k] = L[i];
            i++;
        } else {
            arr[k] = R[j];
            j++;
        }
        k++;
    }

    while (i < n1) {
        arr[k] = L[i];
        i++;
        k++;
    }

    while (j < n2) {
        arr[k] = R[j];
        j++;
        k++;
    }
}

void sequentialMergeSort(std::vector<int>& arr, int l, int r) {
    if (l < r) {
        int m = l + (r - l) / 2;

        sequentialMergeSort(arr, l, m);
        sequentialMergeSort(arr, m + 1, r);

        merge(arr, l, m, r);
    }
}

void parallelMergeSort(std::vector<int>& arr, int l, int r) {
    if (l < r) {
        int m = l + (r - l) / 2;

#pragma omp parallel sections
        {
#pragma omp section
            parallelMergeSort(arr, l, m);
#pragma omp section
            parallelMergeSort(arr, m + 1, r);
        }

        merge(arr, l, m, r);
    }
}

int main() {
    int n = 10000;
    std::vector<int> arr(n), arr_copy(n);

    for (int i = 0; i < n; i++) {
        arr[i] = rand() % 10000;
    }

    for (int i = 0; i < n; i++) {
        arr_copy[i] = arr[i];
    }

    double start_time;

    // Measure time for parallel Merge Sort
    start_time = omp_get_wtime();
    parallelMergeSort(arr, 0, n - 1);
    double parallel_merge_sort_time = omp_get_wtime() - start_time;

    // Measure time for sequential Merge Sort
    start_time = omp_get_wtime();
    sequentialMergeSort(arr_copy, 0, n - 1);
    double sequential_merge_sort_time = omp_get_wtime() - start_time;

    printf("Sequential Merge Sort Time: %f seconds\n", sequential_merge_sort_time);
    printf("Parallel Merge Sort Time: %f seconds\n", parallel_merge_sort_time);

    return 0;
}
