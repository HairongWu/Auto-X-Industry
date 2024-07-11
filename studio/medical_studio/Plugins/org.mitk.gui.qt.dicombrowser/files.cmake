set(SRC_CPP_FILES
)

set(INTERNAL_CPP_FILES
  mitkPluginActivator.cpp
  QmitkDicomBrowser.cpp
  QmitkDicomDirectoryListener.cpp
  QmitkStoreSCPLauncher.cpp
  QmitkStoreSCPLauncherBuilder.cpp
  QmitkDicomDataEventPublisher.cpp
  DicomEventHandler.cpp
  QmitkDicomPreferencePage.cpp
)

set(UI_FILES
  src/internal/QmitkDicomBrowserControls.ui
)

set(MOC_H_FILES
  src/internal/mitkPluginActivator.h
  src/internal/QmitkDicomBrowser.h
  src/internal/QmitkDicomDirectoryListener.h
  src/internal/QmitkStoreSCPLauncher.h
  src/internal/QmitkStoreSCPLauncherBuilder.h
  src/internal/QmitkDicomDataEventPublisher.h
  src/internal/DicomEventHandler.h
  src/internal/QmitkDicomPreferencePage.h
)

# list of resource files which can be used by the plug-in
# system without loading the plug-ins shared library,
# for example the icon used in the menu and tabs for the
# plug-in views in the workbench
set(CACHED_RESOURCE_FILES
  resources/dicom.svg
  plugin.xml
)

# list of Qt .qrc files which contain additional resources
# specific to this plugin
set(QRC_FILES
resources/dicom.qrc
)

set(CPP_FILES )

foreach(file ${SRC_CPP_FILES})
  set(CPP_FILES ${CPP_FILES} src/${file})
endforeach(file ${SRC_CPP_FILES})

foreach(file ${INTERNAL_CPP_FILES})
  set(CPP_FILES ${CPP_FILES} src/internal/${file})
endforeach(file ${INTERNAL_CPP_FILES})

