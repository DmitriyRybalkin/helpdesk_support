#include <iostream>
#include <string>
#include <tensorflow/c/c_api.h>

using namespace std;

void NoOpDeallocator(void *, size_t, void *) {}

int main()
{
    printf("Hello from TensorFlow C library version %s\n", TF_Version());

    TF_Graph *Graph = TF_NewGraph();
    TF_Status *Status = TF_NewStatus();

    TF_SessionOptions *SessionOpts = TF_NewSessionOptions();
    TF_Buffer *RunOpts = NULL;

    const char *saved_model_dir = "cnn_model/"; // Path of the model
    const char *tags = "serve";                 // default model serving tag; can change in future
    int ntags = 1;

    TF_Session *Session = TF_LoadSessionFromSavedModel(SessionOpts,
                                                       RunOpts,
                                                       saved_model_dir,
                                                       &tags,
                                                       ntags,
                                                       Graph,
                                                       NULL,
                                                       Status);
    if (TF_GetCode(Status) == TF_OK) {
        printf("TF_LoadSessionFromSavedModel OK\n");
    } else {
        printf("%s", TF_Message(Status));
    }

    //****** Get input tensor
    int NumInputs = 1;
    TF_Output *Input = (TF_Output *) malloc(sizeof(TF_Output) * NumInputs);

    TF_Output t0 = {TF_GraphOperationByName(Graph, "serving_default_input_1"), 0};
    if (t0.oper == NULL)
        printf("ERROR: Failed TF_GraphOperationByName serving_default_input_1\n");
    else
        printf("TF_GraphOperationByName serving_default_input_1 is OK\n");

    Input[0] = t0;

    //********* Get Output tensor
    int NumOutputs = 1;
    TF_Output *Output = (TF_Output *) malloc(sizeof(TF_Output) * NumOutputs);

    TF_Output t2 = {TF_GraphOperationByName(Graph, "StatefulPartitionedCall"), 0};
    if (t2.oper == NULL)
        printf("ERROR: Failed TF_GraphOperationByName StatefulPartitionedCall\n");
    else
        printf("TF_GraphOperationByName StatefulPartitionedCall is OK\n");

    Output[0] = t2;

    //********* Allocate data for inputs & outputs
    TF_Tensor **InputValues = (TF_Tensor **) malloc(sizeof(TF_Tensor *) * NumInputs);
    TF_Tensor **OutputValues = (TF_Tensor **) malloc(sizeof(TF_Tensor *) * NumOutputs);

    int ndims = 2;
    int64_t dims[] = {1, 279};
    int32_t data[279] = {0};
    memset(data, 0, 279);
    // sentence based on token table of indexed words
    data[0] = 1251;
    data[1] = 5;
    data[2] = 1;
    data[3] = 78;
    data[4] = 1251;
    data[5] = 110;
    data[6] = 4;

    int ndata = sizeof(data); // This is tricky, it number of bytes not number of element

    printf("TF Input data size in bytes: %d\n", ndata);

    TF_Tensor *int_tensor = TF_NewTensor(TF_INT32, dims, ndims, data, ndata, &NoOpDeallocator, 0);
    if (int_tensor != NULL) {
        printf("TF_NewTensor is OK\n");
    } else
        printf("ERROR: Failed TF_NewTensor\n");

    InputValues[0] = int_tensor;

    // //Run the Session
    TF_SessionRun(Session,
                  NULL,
                  Input,
                  InputValues,
                  NumInputs,
                  Output,
                  OutputValues,
                  NumOutputs,
                  NULL,
                  0,
                  NULL,
                  Status);

    if (TF_GetCode(Status) == TF_OK) {
        printf("Session is OK\n");
    } else {
        printf("%s", TF_Message(Status));
    }

    //Free memory
    TF_DeleteGraph(Graph);
    TF_DeleteSession(Session, Status);
    TF_DeleteSessionOptions(SessionOpts);
    TF_DeleteStatus(Status);

    void *buff = TF_TensorData(OutputValues[0]);
    float *offsets = (float *) buff;
    printf("Result Tensor :\n");
    for (int i = 0; i < 5; ++i) {
        printf("%f\n", offsets[i]);
    }

    system("pause");

    return 0;
}
