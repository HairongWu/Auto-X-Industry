/*============================================================================

The Medical Imaging Interaction Toolkit (MITK)

Copyright (c) German Cancer Research Center (DKFZ)
All rights reserved.

Use of this source code is governed by a 3-clause BSD license that can be
found in the LICENSE file.

============================================================================*/

#ifndef mitkItkLoggingAdapter_h
#define mitkItkLoggingAdapter_h

#include <MitkCoreExports.h>
#include <itkObjectFactory.h>
#include <itkOutputWindow.h>

namespace mitk
{
  //##Documentation
  //## @brief Adapter that overwrites the standard itk logging output window and sends the logging messages to the MITK
  // logging instead.
  //## @ingroup IO

  // this class is used to send output to stdout and not the itk window
  class MITKCORE_EXPORT ItkLoggingAdapter : public itk::OutputWindow
  {
  public:
    typedef ItkLoggingAdapter Self;
    typedef itk::SmartPointer<Self> Pointer;
    typedef itk::SmartPointer<const Self> ConstPointer;

    /** Run-time type information (and related methods).   */
    itkTypeMacro(ItkLoggingAdapter, itk::OutputWindow);

    /** New macro for creation of through a Smart Pointer   */
    itkFactorylessNewMacro(Self);
    itkCloneMacro(Self);

      /** @brief Initializes the logging adapter. Itk logging
        *        messages are redirected to MITK logging afterwards.
        */
      static void Initialize();

    void DisplayText(const char *s) override;

  protected:
    ItkLoggingAdapter();
    ~ItkLoggingAdapter() override;

  private:
    ItkLoggingAdapter(const Self &); // purposely not implemented
    void operator=(const Self &);    // purposely not implemented
  };
}

#endif
