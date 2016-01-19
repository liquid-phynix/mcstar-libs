__kernel void increment(__global int* arr, ulong len, ulong iter){
    int i = get_global_id(0);
    int val = arr[i];
    arr[i] = val - len - i;
}

