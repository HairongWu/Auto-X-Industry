set(INTERNAL_CPP_FILES
  mitkPluginActivator.cpp
  QmitkImageCropperView.cpp
  QmitkConvertGeometryDataToROIAction.cpp
)

set(UI_FILES
  src/internal/QmitkImageCropperViewControls.ui
)

set(MOC_H_FILES
  src/internal/mitkPluginActivator.h
  src/internal/QmitkImageCropperView.h
  src/internal/QmitkConvertGeometryDataToROIAction.h
)

set(CACHED_RESOURCE_FILES
  resources/crop.svg
  plugin.xml
)

set(QRC_FILES
  resources/imagecropper.qrc
)

set(CPP_FILES)

foreach(file ${SRC_CPP_FILES})
  set(CPP_FILES ${CPP_FILES} src/${file})
endforeach(file ${SRC_CPP_FILES})

foreach(file ${INTERNAL_CPP_FILES})
  set(CPP_FILES ${CPP_FILES} src/internal/${file})
endforeach(file ${INTERNAL_CPP_FILES})
