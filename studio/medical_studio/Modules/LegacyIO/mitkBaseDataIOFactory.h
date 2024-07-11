/*============================================================================

The Medical Imaging Interaction Toolkit (MITK)

Copyright (c) German Cancer Research Center (DKFZ)
All rights reserved.

Use of this source code is governed by a 3-clause BSD license that can be
found in the LICENSE file.

============================================================================*/

#ifndef mitkBaseDataIOFactory_h
#define mitkBaseDataIOFactory_h

#include <MitkLegacyIOExports.h>

#include "mitkBaseData.h"

#include "itkObject.h"

namespace mitk
{
  /**
   * @brief BaseDataIO creates instances of BaseData objects using an object factory.
   *
   * @ingroup MitkLegacyIOModule
   * @deprecatedSince{2014_10} Use mitk::IOUtils or mitk::FileReaderRegistry instead.
   */
  class DEPRECATED() MITKLEGACYIO_EXPORT BaseDataIO : public itk::Object
  {
  public:
    /** Standard class typedefs. */
    typedef BaseDataIO Self;
    typedef itk::Object Superclass;
    typedef itk::SmartPointer<Self> Pointer;
    typedef itk::SmartPointer<const Self> ConstPointer;

    /** Class Methods used to interface with the registered factories */

    /** Run-time type information (and related methods). */
    itkTypeMacro(BaseDataIO, Object);

    /** Create the appropriate BaseData depending on the particulars of the file. */
    static std::vector<mitk::BaseData::Pointer> LoadBaseDataFromFile(const std::string path,
                                                                     const std::string filePrefix,
                                                                     const std::string filePattern,
                                                                     bool series);

  protected:
    BaseDataIO();
    ~BaseDataIO() override;

  private:
    BaseDataIO(const Self &);     // purposely not implemented
    void operator=(const Self &); // purposely not implemented
  };

} // end namespace mitk

#endif
