/*============================================================================

The Medical Imaging Interaction Toolkit (MITK)

Copyright (c) German Cancer Research Center (DKFZ)
All rights reserved.

Use of this source code is governed by a 3-clause BSD license that can be
found in the LICENSE file.

============================================================================*/

#ifndef mitkSinglePointDataInteractor_h
#define mitkSinglePointDataInteractor_h

#include "itkObject.h"
#include "itkObjectFactory.h"
#include "itkSmartPointer.h"
#include "mitkCommon.h"
#include "mitkPointSetDataInteractor.h"
#include <MitkCoreExports.h>
#include <mitkPointSet.h>

namespace mitk
{
  /**
   * Class SinglePointDataInteractor
   * \brief Implementation of the single point interaction
   *
   * Interactor operates on a single point set, when a data node is set, its containing point set is clear for
   * initialization.
   */

  // Inherit from DataInteratcor, this provides functionality of a state machine and configurable inputs.
  class MITKCORE_EXPORT SinglePointDataInteractor : public PointSetDataInteractor
  {
  public:
    mitkClassMacro(SinglePointDataInteractor, PointSetDataInteractor);
    itkFactorylessNewMacro(Self);
    itkCloneMacro(Self);

      protected : SinglePointDataInteractor();
    ~SinglePointDataInteractor() override;

    /** Adds a point at the given coordinates.
     *  This function overwrites the behavior of PointSetDataInteractor such that instead of adding new points
     *  the first points position is updated. All other interaction (move,delete) is still handled by
     * PointSetDataInteractor.
     */
    void AddPoint(StateMachineAction *, InteractionEvent *event) override;

    /**
     * @brief SetMaxPoints Sets the maximal number of points for the pointset
     * Overwritten, per design this class will always have a maximal number of one.
     * @param maxNumber
     */
    virtual void SetMaxPoints(unsigned int maxNumber = 0);
    void DataNodeChanged() override;
  };
}
#endif
