workspace(name = "opencv_bazel_camera")

new_local_repository(
    name = "opencv_local",
    path = "/opt/homebrew/opt/opencv",
    build_file_content = """
cc_library(
    name = \"opencv\",
    hdrs = glob([\"include/**\"]),
    srcs = [],
    includes = [\"include/opencv4\"],
    linkstatic = 0,
    linkopts = [
        \"-L/opt/homebrew/opt/opencv/lib\",
        \"-lopencv_core\",
        \"-lopencv_imgproc\",
        \"-lopencv_highgui\",
        \"-lopencv_videoio\",
        \"-lopencv_imgcodecs\",
        \"-lopencv_objdetect\",
        \"-framework AVFoundation\",
        \"-framework CoreMedia\",
        \"-framework CoreVideo\",
    ],
    visibility = [\"//visibility:public\"],
)
""",
)