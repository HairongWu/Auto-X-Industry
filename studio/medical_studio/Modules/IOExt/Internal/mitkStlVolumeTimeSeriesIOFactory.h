/*============================================================================

The Medical Imaging Interaction Toolkit (MITK)

Copyright (c) German Cancer Research Center (DKFZ)
All rights reserved.

Use of this source code is governed by a 3-clause BSD license that can be
found in the LICENSE file.

============================================================================*/
#ifndef mitkStlVolumeTimeSeriesIOFactory_h
#define mitkStlVolumeTimeSeriesIOFactory_h

#ifdef _MSC_VER
#pragma warning(disable : 4786)
#endif

#include "itkObjectFactoryBase.h"
#include "mitkBaseData.h"

namespace mitk
{
  //##Documentation
  //## @brief Create instances of StlVolumeTimeSeriesReader objects using an object factory.
  class StlVolumeTimeSeriesIOFactory : public itk::ObjectFactoryBase
  {
  public:
    /** Standard class typedefs. */
    typedef StlVolumeTimeSeriesIOFactory Self;
    typedef itk::ObjectFactoryBase Superclass;
    typedef itk::SmartPointer<Self> Pointer;
    typedef itk::SmartPointer<const Self> ConstPointer;

    /** Class methods used to interface with the registered factories. */
    const char *GetITKSourceVersion(void) const override;
    const char *GetDescription(void) const override;

    /** Method for class instantiation. */
    itkFactorylessNewMacro(Self);
    static StlVolumeTimeSeriesIOFactory *FactoryNew() { return new StlVolumeTimeSeriesIOFactory; }
    /** Run-time type information (and related methods). */
    itkTypeMacro(StlVolumeTimeSeriesIOFactory, ObjectFactoryBase);

    /**
     * Register one factory of this type
     * \deprecatedSince{2013_09}
     */
    DEPRECATED(static void RegisterOneFactory(void))
    {
      StlVolumeTimeSeriesIOFactory::Pointer StlVolumeTimeSeriesIOFactory = StlVolumeTimeSeriesIOFactory::New();
      ObjectFactoryBase::RegisterFactory(StlVolumeTimeSeriesIOFactory);
    }

  protected:
    StlVolumeTimeSeriesIOFactory();
    ~StlVolumeTimeSeriesIOFactory() override;

  private:
    StlVolumeTimeSeriesIOFactory(const Self &); // purposely not implemented
    void operator=(const Self &);               // purposely not implemented
  };

} // end namespace mitk

#endif
