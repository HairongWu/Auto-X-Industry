/*============================================================================

The Medical Imaging Interaction Toolkit (MITK)

Copyright (c) German Cancer Research Center (DKFZ)
All rights reserved.

Use of this source code is governed by a 3-clause BSD license that can be
found in the LICENSE file.

============================================================================*/

#ifndef mitkCoreObjectFactoryBase_h
#define mitkCoreObjectFactoryBase_h

// the mitkLog.h header is necessary for CMake test drivers.
// Since the EXTRA_INCLUDE parameter of CREATE_TEST_SOURCELIST only
// allows one extra include file, we specify mitkLog.h here so it will
// be available to all classes implementing this interface.
#include <mitkLog.h>

#include "mitkFileWriterWithInformation.h"
#include "mitkMapper.h"
#include <MitkCoreExports.h>
#include <itkObjectFactoryBase.h>
#include <itkVersion.h>

namespace mitk
{
  class DataNode;

  //## @brief base-class for factories of certain mitk objects.
  //## @ingroup Algorithms
  //## This interface can be implemented by factories which add new mapper classes or extend the
  //## data tree deserialization mechanism.

  class MITKCORE_EXPORT CoreObjectFactoryBase : public itk::Object
  {
  public:
    typedef std::list<mitk::FileWriterWithInformation::Pointer> FileWriterList;
    typedef std::multimap<std::string, std::string> MultimapType;

    mitkClassMacroItkParent(CoreObjectFactoryBase, itk::Object);

      virtual Mapper::Pointer CreateMapper(mitk::DataNode *node, MapperSlotId slotId) = 0;
    virtual void SetDefaultProperties(mitk::DataNode *node) = 0;

    /**
     * @deprecatedSince{2014_10} See mitk::FileReaderRegistry and QmitkIOUtil
     */
    virtual std::string GetFileExtensions() = 0;

    /**
     * @deprecatedSince{2014_10} See mitk::FileReaderRegistry and QmitkIOUtil
     */
    virtual MultimapType GetFileExtensionsMap() = 0;

    /**
     * @deprecatedSince{2014_10} See mitk::FileWriterRegistry and QmitkIOUtil
     */
    virtual std::string GetSaveFileExtensions() = 0;

    /**
     * @deprecatedSince{2014_10} See mitk::FileWriterRegistry and QmitkIOUtil
     */
    virtual MultimapType GetSaveFileExtensionsMap() = 0;

    virtual const char *GetITKSourceVersion() const { return ITK_SOURCE_VERSION; }
    virtual const char *GetDescription() const { return "Core Object Factory"; }
    /**
     * @deprecatedSince{2014_10} See mitk::FileWriterRegistry
     */
    FileWriterList GetFileWriters() { return m_FileWriters; }
  protected:
    /**
     * @brief create a string from a map that contains the file extensions
     * @param fileExtensionsMap input map with the file extensions, e.g. ("*.dcm", "DICOM files")("*.dc3", "DICOM
     * files")
     * @param fileExtensions the converted output string, suitable for the QT QFileDialog widget
     *                       e.g. "all (*.dcm *.DCM *.dc3 ... *.vti *.hdr *.nrrd *.nhdr );;ODF Images (*.odf *qbi)"
     *
     * @deprecatedSince{2014_10}
     */
    static void CreateFileExtensions(MultimapType fileExtensionsMap, std::string &fileExtensions);

    FileWriterList m_FileWriters;

    friend class CoreObjectFactory;
  };
}
#endif
