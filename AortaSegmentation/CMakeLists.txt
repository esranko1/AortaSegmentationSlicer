cmake_minimum_required(VERSION 3.20.6...3.22.6 FATAL_ERROR)

project(AortaSegmentation)

#-----------------------------------------------------------------------------
# Extension meta-information
set(EXTENSION_HOMEPAGE "https://www.slicer.org/wiki/Documentation/Nightly/Extensions/AortaSegmentation")
set(EXTENSION_CONTRIBUTORS "Eszter Sranko (JHU)")
set(EXTENSION_DESCRIPTION "MRI aorta segmentation.")
set(EXTENSION_ICONURL "https://www.example.com/Slicer/Extensions/AortaSegmentation.png")
set(EXTENSION_SCREENSHOTURLS "https://www.example.com/Slicer/Extensions/AortaSegmentation/Screenshots/1.png")
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
