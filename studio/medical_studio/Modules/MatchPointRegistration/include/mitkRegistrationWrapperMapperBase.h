/*============================================================================

The Medical Imaging Interaction Toolkit (MITK)

Copyright (c) German Cancer Research Center (DKFZ)
All rights reserved.

Use of this source code is governed by a 3-clause BSD license that can be
found in the LICENSE file.

============================================================================*/


#ifndef mitkRegistrationWrapperMapperBase_h
#define mitkRegistrationWrapperMapperBase_h


#include <vtkSmartPointer.h>

#include <mitkVtkMapper.h>
#include <mitkLocalStorageHandler.h>

#include "MitkMatchPointRegistrationExports.h"

class vtkPropAssembly;
class vtkPolyDataMapper;
class vtkPolyData;
class vtkColorTransferFunction;
class vtkActor;

namespace mitk {

/**Base class for all mapper that visualize a registration object.*/
class MITKRegistrationWrapperMapperBase : public VtkMapper
{
public:

    mitkClassMacro(MITKRegistrationWrapperMapperBase, VtkMapper);

    //========== essential implementation for mapper ==========
    vtkProp *GetVtkProp(mitk::BaseRenderer *renderer) override;
    static void SetDefaultProperties(DataNode* node, BaseRenderer* renderer = nullptr, bool overwrite = false );
    void GenerateDataForRenderer(mitk::BaseRenderer* renderer) override;
    //=========================================================

    virtual bool GetGeometryDescription(mitk::BaseRenderer *renderer, mitk::BaseGeometry::ConstPointer& gridDesc, unsigned int& gridFrequ) const = 0;
    virtual bool RendererGeometryIsOutdated(mitk::BaseRenderer *renderer, const itk::TimeStamp& time) const = 0;

    /**Internal class to store all informations and helper needed by a mapper to provide the render data for a
     certain renderer.*/
    class MITKMATCHPOINTREGISTRATION_EXPORT RegWrapperLocalStorage : public mitk::Mapper::BaseLocalStorage
    {
    public:
      vtkSmartPointer<vtkPolyData> m_DeformedGridData;
      vtkSmartPointer<vtkPolyData> m_StartGridData;

      vtkSmartPointer<vtkActor> m_DeformedGridActor;
      vtkSmartPointer<vtkPolyDataMapper> m_DeformedGridMapper;
      vtkSmartPointer<vtkActor> m_StartGridActor;
      vtkSmartPointer<vtkPolyDataMapper> m_StartGridMapper;

      vtkSmartPointer<vtkPropAssembly> m_RegAssembly;

      vtkSmartPointer<vtkColorTransferFunction>  m_LUT;

        /** \brief Timestamp of last update of stored data. */
        itk::TimeStamp m_LastUpdateTime;
        /** \brief Constructor of the local storage. Do as much actions as possible in here to avoid double executions. */
        RegWrapperLocalStorage();

        ~RegWrapperLocalStorage() override
        {
        }
    };

    /** \brief This member holds all three LocalStorages for the 3D render window(s). */
    mitk::LocalStorageHandler<RegWrapperLocalStorage> m_LSH;


protected:

    MITKRegistrationWrapperMapperBase();
    ~MITKRegistrationWrapperMapperBase() override;


private:

};

} // end namespace mitk




#endif
