set(SOURCE_FILES
    "memory.h"
    "memory.cpp"
    "random.cpp"
    "random.h"
    "ThreeD.h"
    "ThreeD.cpp"
	"md.h"
	"md.cu"
    )

cuda_add_library(src
    ${SOURCE_FILES}
    OPTIONS -arch=sm_20
    )
