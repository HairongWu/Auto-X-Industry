/*============================================================================

The Medical Imaging Interaction Toolkit (MITK)

Copyright (c) German Cancer Research Center (DKFZ)
All rights reserved.

Use of this source code is governed by a 3-clause BSD license that can be
found in the LICENSE file.

============================================================================*/
#ifndef mitkIOAdapter_h
#define mitkIOAdapter_h

#include "mitkBaseProcess.h"

#include "itkObject.h"

namespace mitk
{
  /**
   * @brief IOAdapterBase class is an abstract adapter class for IO process objects.
   *
   * @ingroup DeprecatedIO
   * @deprecatedSince{2014_10} Use mitk::IFileReader instead
   */
  class IOAdapterBase : public itk::Object
  {
  public:
    /** Standard typedefs. */
    typedef IOAdapterBase Self;
    typedef itk::Object Superclass;
    typedef itk::SmartPointer<Self> Pointer;
    typedef itk::SmartPointer<const Self> ConstPointer;

    /// Create an object and return a pointer to it as a mitk::BaseProcess.
    virtual itk::SmartPointer<BaseDataSource> CreateIOProcessObject(const std::string filename,
                                                                    const std::string filePrefix,
                                                                    const std::string filePattern) = 0;
    virtual bool CanReadFile(const std::string filename,
                             const std::string filePrefix,
                             const std::string filePattern) = 0;

  protected:
    IOAdapterBase() {}
    ~IOAdapterBase() override {}
  private:
    IOAdapterBase(const Self &);  // purposely not implemented
    void operator=(const Self &); // purposely not implemented
  };

  /**
   * @brief IOAdapter class is an adapter class for instantiation of IO process objects.
   * Additional this interface defines the function CanReadFile().
   * This interface allows the target (object) the access to the adaptee (IO process object).
   *
   * @ingroup IO
   * @deprecatedSince{2014_10} Use mitk::IFileReader instead
   */
  template <class T>
  class IOAdapter : public IOAdapterBase
  {
  public:
    /** Standard class typedefs. */
    typedef IOAdapter Self;
    typedef itk::SmartPointer<Self> Pointer;

    /** Methods from mitk::BaseProcess. */
    itkFactorylessNewMacro(Self);
    mitk::BaseDataSource::Pointer CreateIOProcessObject(const std::string filename,
                                                        const std::string filePrefix,
                                                        const std::string filePattern) override
    {
      typename T::Pointer ioProcessObject = T::New();
      ioProcessObject->SetFileName(filename.c_str());
      ioProcessObject->SetFilePrefix(filePrefix.c_str());
      ioProcessObject->SetFilePattern(filePattern.c_str());
      return ioProcessObject.GetPointer();
    }

    bool CanReadFile(const std::string filename,
                             const std::string filePrefix,
                             const std::string filePattern) override
    {
      return T::CanReadFile(filename, filePrefix, filePattern);
    }

  protected:
    IOAdapter() {}
    ~IOAdapter() override {}
  private:
    IOAdapter(const Self &);      // purposely not implemented
    void operator=(const Self &); // purposely not implemented
  };

} // end namespace mitk

#endif
