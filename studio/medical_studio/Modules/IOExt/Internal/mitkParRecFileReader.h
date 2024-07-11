/*============================================================================

The Medical Imaging Interaction Toolkit (MITK)

Copyright (c) German Cancer Research Center (DKFZ)
All rights reserved.

Use of this source code is governed by a 3-clause BSD license that can be
found in the LICENSE file.

============================================================================*/

#ifndef mitkParRecFileReader_h
#define mitkParRecFileReader_h

#include "mitkCommon.h"
#include "mitkFileReader.h"
#include "mitkImageSource.h"

namespace mitk
{
  //##Documentation
  //## @brief Reader to read files in Philips PAR/REC file format
  class ParRecFileReader : public ImageSource, public FileReader
  {
  public:
    mitkClassMacro(ParRecFileReader, FileReader);

    /** Method for creation through the object factory. */
    itkFactorylessNewMacro(Self);
    itkCloneMacro(Self);
    itkSetStringMacro(FileName);
    itkGetStringMacro(FileName);

    itkSetStringMacro(FilePrefix);
    itkGetStringMacro(FilePrefix);

    itkSetStringMacro(FilePattern);
    itkGetStringMacro(FilePattern);

    static bool CanReadFile(const std::string filename, const std::string filePrefix, const std::string filePattern);

  protected:
    void GenerateData() override;

    void GenerateOutputInformation() override;

    ParRecFileReader();

    ~ParRecFileReader() override;

    //##Description
    //## @brief Time when Header was last read
    itk::TimeStamp m_ReadHeaderTime;

  protected:
    std::string m_FileName;
    std::string m_RecFileName;

    std::string m_FilePrefix;

    std::string m_FilePattern;
  };

} // namespace mitk

#endif
