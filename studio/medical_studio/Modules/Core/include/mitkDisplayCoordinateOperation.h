/*============================================================================

The Medical Imaging Interaction Toolkit (MITK)

Copyright (c) German Cancer Research Center (DKFZ)
All rights reserved.

Use of this source code is governed by a 3-clause BSD license that can be
found in the LICENSE file.

============================================================================*/

#ifndef mitkDisplayCoordinateOperation_h
#define mitkDisplayCoordinateOperation_h

#include "mitkBaseRenderer.h"
#include "mitkNumericTypes.h"
#include "mitkOperation.h"
#include <MitkCoreExports.h>
#include <mitkWeakPointer.h>

#define mitkGetMacro(name, type)                                                                                       \
  virtual type Get##name() { return this->m_##name; }
namespace mitk
{
  // TODO Legacy , no longer necessary when after migrating all DisplayInteractions to new Interactions.
  // Coordinate supplier can probably also be removed then.

  //##Documentation
  //## @brief Operation with information necessary for operations of DisplayVectorInteractor
  //## @ingroup Undo
  class MITKCORE_EXPORT DisplayCoordinateOperation : public Operation
  {
  public:
    DisplayCoordinateOperation(mitk::OperationType operationType,
                               mitk::BaseRenderer *renderer,
                               const mitk::Point2D &startDisplayCoordinate,
                               const mitk::Point2D &lastDisplayCoordinate,
                               const mitk::Point2D &currentDisplayCoordinate);

    DisplayCoordinateOperation(mitk::OperationType operationType,
                               mitk::BaseRenderer *renderer,
                               const mitk::Point2D &startDisplayCoordinate,
                               const mitk::Point2D &lastDisplayCoordinate,
                               const mitk::Point2D &currentDisplayCoordinate,
                               const mitk::Point2D &startCoordinateInMM);

    ~DisplayCoordinateOperation() override;

    mitk::BaseRenderer *GetRenderer();

    mitkGetMacro(StartDisplayCoordinate, mitk::Point2D);
    mitkGetMacro(LastDisplayCoordinate, mitk::Point2D);
    mitkGetMacro(CurrentDisplayCoordinate, mitk::Point2D);
    mitkGetMacro(StartCoordinateInMM, mitk::Point2D);

    mitk::Vector2D GetLastToCurrentDisplayVector();
    mitk::Vector2D GetStartToCurrentDisplayVector();
    mitk::Vector2D GetStartToLastDisplayVector();

  private:
    mitk::WeakPointer<mitk::BaseRenderer> m_Renderer;

    const mitk::Point2D m_StartDisplayCoordinate;
    const mitk::Point2D m_LastDisplayCoordinate;
    const mitk::Point2D m_CurrentDisplayCoordinate;
    const mitk::Point2D m_StartCoordinateInMM;
  };
}

#endif
