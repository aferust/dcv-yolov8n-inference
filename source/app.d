
import std.stdio;
import std.process;
import std.datetime.stopwatch : StopWatch;
import std.conv : to;

import dcv.core;
import dcv.imgproc;
import dcv.plot;
import dcv.imageio;

import onnxruntime; // use importC

import mir.ndslice, mir.rc;
import mir.appender;

void main()
{
    auto yolov8ninfer = ORTContextYOLOV8N("yolov8n.onnx" /*, OrtLoggingLevel.ORT_LOGGING_LEVEL_VERBOSE*/);

    auto font = TtfFont(cast(ubyte[])import("Nunito-Regular.ttf"));
    
    Image img = imread("cat_dog.jpg");
    scope(exit) destroyFree(img);

    Slice!(ubyte*, 3) imSlice = img.sliced; // will be freed with the previous destroyFree(img)

    auto impr = letterBoxAndPreprocess(imSlice);//preprocess(imSlice);

    scope float* outPtr;
    long[3] outDims;
    size_t numberOfelements;
    // foreach(i; 0..50) // if you use a GPU backend warm up the GPU before entering into any video loop
    yolov8ninfer.infer(impr, outPtr, outDims, numberOfelements);

    scope Slice!(float*, 3) outSlice = outPtr[0..numberOfelements].sliced(outDims[0], outDims[1], outDims[2]); 

    scope boxes = extractBoxCoordinates(outSlice, 0.50f).data;
    
    // - yolov8 yields multiple detections for the same object. We can consolidate the detections
    //   for the same object by computing distances between the box centroids. This situation can be observed
    //   on the cat_dog.jpg, since the dog detected multiple times.
    //   reaad for a post processing approach https://github.com/ultralytics/ultralytics/issues/5811#issuecomment-1771565130
    foreach(box; boxes){
        putRectangleHollow(imSlice, [APoint(box[0], box[1]), APoint(box[2], box[3])], aBlue, 2);
        putRectangleSolid(imSlice, [APoint(box[0], box[1]), APoint(box[2], box[3])], [255,0,0,10]);
        putText(imSlice, font, classNames[cast(ulong)box[4]], APoint(cast(float)box[0], cast(float)box[1]), 30, aGreen);

    }
    
    //imshow((impr.transposed!(1, 2, 0) * 255).rcslice, "detection"); // letterbox image that we feed to the network

    imshow(imSlice, "detection"); // show the original image and use the scale for box coords
    imwrite(imSlice, ImageFormat.IF_RGB, "cat_dog_result.jpg"); // write the result on the disk

    waitKey();   
}

struct ORTContextYOLOV8N {

    private {
        const(OrtApi)* ort;
        OrtEnv* env;
        OrtSessionOptions* session_options;
        OrtSession* session;
        OrtMemoryInfo* memory_info;
        OrtValue*[1] output_tensors;
        const(char)*[1] input_node_names;
        const(char)*[1] output_node_names;
    }
    
    this(in wchar[] networkFilePath, OrtLoggingLevel LOGlevel = OrtLoggingLevel.ORT_LOGGING_LEVEL_ERROR)
    {
        // slower with : OrtLoggingLevel.ORT_LOGGING_LEVEL_VERBOSE
        
        ort = OrtGetApiBase().GetApi(ORT_API_VERSION);
        assert(ort);
        
        checkStatus(ort.CreateEnv(LOGlevel, "test", &env));
        checkStatus(ort.CreateSessionOptions(&session_options));
        ort.SetIntraOpNumThreads(session_options, 4);
        ort.SetSessionLogSeverityLevel(session_options, 4);

        ort.SetSessionGraphOptimizationLevel(session_options, GraphOptimizationLevel.ORT_ENABLE_ALL);
        ort.SetSessionExecutionMode(session_options, ExecutionMode.ORT_PARALLEL);

        // with a proper linkage you can use hardware accel with different backends
        //checkStatus(OrtSessionOptionsAppendExecutionProvider_CUDA(session_options, 0));
        //checkStatus(OrtSessionOptionsAppendExecutionProvider_CPU(session_options, 0));

        checkStatus(ort.CreateSession(env, cast(const(ushort)*)networkFilePath.ptr, session_options, &session));

        OrtAllocator* allocator; // https://onnxruntime.ai/docs/api/c/struct_ort_api.html#a8dec797ae52ee1a681e4f88be1fb4bb3
        checkStatus(ort.GetAllocatorWithDefaultOptions(&allocator)); // allocator should NOT be freed according to docs

        size_t num_input_nodes;
        // get number of model input nodes
        checkStatus(ort.SessionGetInputCount(session, &num_input_nodes));

        input_node_names = ["images".ptr];
        
        { // We can infer input image shape from the network file
            OrtTypeInfo* input_type_info;
            ort.SessionGetInputTypeInfo(session, 0, &input_type_info);

            // Get input node shape
            OrtTensorTypeAndShapeInfo* input_shape_info;
            ort.CastTypeInfoToTensorInfo(input_type_info, &input_shape_info);

            size_t num_dims;
            ort.GetDimensionsCount(input_shape_info, &num_dims);
            long* input_dims = cast(long*)malloc(num_dims * long.sizeof);
            ort.GetDimensions(input_shape_info, input_dims, num_dims);

            // Output input image size
            printf("Input image size: Height=%lld, Width=%lld, Channels=%lld\n", 
                input_dims[1], input_dims[2], input_dims[3]);
            
            free(input_dims);
        }
        checkStatus(ort.CreateCpuMemoryInfo(OrtAllocatorType.OrtArenaAllocator,
            OrtMemType.OrtMemTypeDefault, &memory_info));
        
        // output tensor
        output_node_names = ["output0"]; // check those : https://netron.app/
    }

    ~this(){
        _free();
    }
    
    private void _free(){
        freeOutInfer();
        ort.ReleaseEnv(env);
        ort.ReleaseSessionOptions(session_options);
        ort.ReleaseSession(session);
        ort.ReleaseMemoryInfo(memory_info);
    }

    void checkStatus(OrtStatus* status)
    {
        import core.stdc.stdlib : exit;
        if (status)
        {
            auto msg = ort.GetErrorMessage(status);
            printf("%s\n", msg);
            ort.ReleaseStatus(status);
            exit(-1); // throw an exception if you run this on a server
        }
    }

    /++
        outPtr will be freed on a call freeOutInfer()
        this is ugly but no extra copy
    +/
    void infer(InputSlice)(auto ref InputSlice impr, out float* outPtr, out long[3] outDims, out size_t ecount0)
    {
        // outDims for yolov8 should be 1 84 8400.

        import core.stdc.stdlib : malloc, free;

        freeOutInfer();

        OrtValue*[1] input_tensor;

        long[4] in1 = [1, 3, 640, 640];
        size_t input_tensor_size = 640 * 640 * 3;
        checkStatus(ort.CreateTensorWithDataAsOrtValue(
            memory_info, cast(void*)impr.ptr, input_tensor_size * float.sizeof, in1.ptr, 4,
                ONNXTensorElementDataType.ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &input_tensor[0]));

        scope (exit)
            ort.ReleaseValue(input_tensor[0]);

        int is_tensor;
        checkStatus(ort.IsTensor(input_tensor[0], &is_tensor));
        assert(is_tensor);

        
        checkStatus(ort.Run(session, null, input_node_names.ptr, input_tensor.ptr, 1,
                output_node_names.ptr, 1, output_tensors.ptr));

        checkStatus(ort.GetTensorMutableData(output_tensors[0], cast(void**)&outPtr));
        checkStatus(ort.IsTensor(output_tensors[0], &is_tensor));
        assert(is_tensor);

        OrtTensorTypeAndShapeInfo* sh0; scope(exit) ort.ReleaseTensorTypeAndShapeInfo(sh0);
        checkStatus(ort.GetTensorTypeAndShape(output_tensors[0], &sh0));
        
        checkStatus(ort.GetTensorShapeElementCount(sh0, &ecount0));
        size_t dcount0;
        checkStatus(ort.GetDimensionsCount(sh0, &dcount0));
        long[] dims0 = (cast(long*)malloc(dcount0 * long.sizeof))[0..dcount0];
        checkStatus(ort.GetDimensions(sh0, dims0.ptr, dcount0));

        outDims = [dims0[0], dims0[1], dims0[2]];

        free(cast(void*)dims0.ptr);
    }
    
    private
    /++
        must be called after each call of infer
    +/
    void freeOutInfer(){
        if(output_tensors[0] !is null){
            ort.ReleaseValue(output_tensors[0]);
            output_tensors[0] = null;
        }
    }
}

float scale;

@nogc nothrow:
/+
auto letterbox_image(InputImg)(InputImg image, size_t h, size_t w){
    import std.algorithm.comparison : min;

    auto iw = image.shape[1];
    auto ih = image.shape[0];
    scale = min((cast(float)w)/iw, (cast(float)h)/ih);
    auto nw = cast(int)(iw*scale);
    auto nh = cast(int)(ih*scale);
    
    auto resized = resize(image, [nh, nw]);
    
    auto new_image = rcslice!ubyte([h, w, 3], 128);
    new_image[0..nh, 0..nw, 0..$] = resized[0..nh, 0..nw, 0..$];
    
    //imshow(new_image, ImageFormat.IF_RGB); waitKey();
    
    return new_image;
}

auto preprocess(InputImg)(InputImg img){
    auto w = 640;
    auto h = 640;
    auto boxed_image = letterbox_image(img, h, w);
    auto image_data = boxed_image.as!float;
    
    auto image_data_t = (image_data / 255.0f).transposed!(2, 0, 1);
    

    //auto blob = slice!float([1, 3, 640, 640], 0);
    //blob[0, 0..3, 0..640, 0..640] = image_data_t[0..3, 0..640, 0..640];

    //return blob;
    return image_data_t.rcslice;
}
+/

Slice!(RCI!float, 3) letterBoxAndPreprocess(InputSlice)(InputSlice img){
    import std.algorithm.comparison : min;

    static assert(InputSlice.N == 3, "only RGB color images are supported");

    size_t w = 640;
    size_t h = 640;

    auto iw = img.shape[1];
    auto ih = img.shape[0];
    scale = min((cast(float)w)/iw, (cast(float)h)/ih);
    auto nw = cast(int)(iw*scale);
    auto nh = cast(int)(ih*scale);
    
    auto resized = resize(img, [nh, nw]);
    
    auto boxed_image = rcslice!float([h, w, 3], 128.0f); // allocates
    boxed_image[0..nh, 0..nw, 0..$] = resized[0..nh, 0..nw, 0..$].as!float; // assign values from a lazy iter 
    
    auto image_data_t = (boxed_image / 255.0f).transposed!(2, 0, 1); // lazy

    return image_data_t.rcslice; // allocates from the lazy slice

}

scope auto extractBoxCoordinates(S)(auto ref S outSlice, float confidenceThreshold) {
    import mir.algorithm.iteration : minIndex, maxIndex;
    import std.array : staticArray;
    import core.lifetime : move;

    auto boxes = scopedBuffer!(float[6]);

    foreach (i; 0 .. outSlice.shape[2]) {
        auto classProbabilities = outSlice[0, 4 .. $, i];

        //auto minClassLoc = classProbabilities.minIndex[0];
        auto maxClassLoc = classProbabilities.maxIndex[0];
        //auto minScore = classProbabilities[minClassLoc];
        auto maxScore = classProbabilities[maxClassLoc];

        if (maxScore > confidenceThreshold) {
            // Object detected with confidence higher than the threshold
            // Extract bounding box coordinates
            auto width = outSlice[0, 2, i]  ;
            auto height = outSlice[0, 3, i] ;

            auto x = outSlice[0, 0, i] - 0.5f * width;
            auto y = outSlice[0, 1, i] - 0.5f * height;
            
            // only one scale value is enough with a letterbox image.
            boxes.put([x/scale, y/scale, (x+width)/scale, (y+height)/scale, maxClassLoc, maxScore].staticArray);
        }
    }
    return boxes.move;
}

enum classNames = [
"person",
"bicycle",
"car",
"motorbike",
"aeroplane",
"bus",
"train",
"truck",
"boat",
"traffic light",
"fire hydrant",
"stop sign",
"parking meter",
"bench",
"bird",
"cat",
"dog",
"horse",
"sheep",
"cow",
"elephant",
"bear",
"zebra",
"giraffe",
"backpack",
"umbrella",
"handbag",
"tie",
"suitcase",
"frisbee",
"skis",
"snowboard",
"sports ball",
"kite",
"baseball bat",
"baseball glove",
"skateboard",
"surfboard",
"tennis racket",
"bottle",
"wine glass",
"cup",
"fork",
"knife",
"spoon",
"bowl",
"banana",
"apple",
"sandwich",
"orange",
"broccoli",
"carrot",
"hot dog",
"pizza",
"donut",
"cake",
"chair",
"sofa",
"pottedplant",
"bed",
"diningtable",
"toilet",
"tvmonitor",
"laptop",
"mouse",
"remote",
"keyboard",
"cell phone",
"microwave",
"oven",
"toaster",
"sink",
"refrigerator",
"book",
"clock",
"vase",
"scissors",
"teddy bear",
"hair drier",
"toothbrush"
];