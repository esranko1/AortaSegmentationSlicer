cmake_minimum_required(VERSION 3.20.6...3.22.6 FATAL_ERROR)

project(AortaSegmentation)

#-----------------------------------------------------------------------------
# Extension meta-information
set(EXTENSION_HOMEPAGE "https://github.com/esranko1/AortaSegmentationSlicer")
set(EXTENSION_CONTRIBUTORS "Eszter Sranko (JHU)")
set(EXTENSION_DESCRIPTION "Segments the aorta from a single MRI volume using a pretrained nnU-Net v2 model.")
set(EXTENSION_ICONURL "https://raw.githubusercontent.com/esranko1/AortaSegmentationSlicer/main/AortaSegmentation/Resources/Icons/AortaSegmentation.png")
set(EXTENSION_SCREENSHOTURLS "https://raw.githubusercontent.com/esranko1/AortaSegmentationSlicer/main/Screenshots/1.png")
set(EXTENSION_DEPENDS "NA") # Specified as a list or "NA" if no dependencies

#-----------------------------------------------------------------------------
# Extension dependencies
find_package(Slicer REQUIRED)
include(${Slicer_USE_FILE})

#-----------------------------------------------------------------------------
# Extension modules
add_subdirectory(AortaSegmentation)
## NEXT_MODULE

#-----------------------------------------------------------------------------
include(${Slicer_EXTENSION_GENERATE_CONFIG})
include(${Slicer_EXTENSION_CPACK})
