/*============================================================================

The Medical Imaging Interaction Toolkit (MITK)

Copyright (c) German Cancer Research Center (DKFZ)
All rights reserved.

Use of this source code is governed by a 3-clause BSD license that can be
found in the LICENSE file.

============================================================================*/

#ifndef mitkProgressBarImplementation_h
#define mitkProgressBarImplementation_h
#include <MitkCoreExports.h>

namespace mitk
{
  //##Documentation
  //## @brief GUI independent Interface for all Gui dependent implementations of a ProgressBar.
  class MITKCORE_EXPORT ProgressBarImplementation
  {
  public:
    //##Documentation
    //## @brief Constructor
    ProgressBarImplementation(){};

    //##Documentation
    //## @brief Destructor
    virtual ~ProgressBarImplementation(){};

    //##Documentation
    //## @brief Sets whether the current progress value is displayed.
    virtual void SetPercentageVisible(bool visible) = 0;

    //##Documentation
    //## @brief Explicitly reset progress bar.
    virtual void Reset() = 0;

    //##Documentation
    //## @brief Adds steps to totalSteps.
    virtual void AddStepsToDo(unsigned int steps) = 0;

    //##Documentation
    //## @brief Sets the current amount of progress to current progress + steps.
    //## @param steps the number of steps done since last Progress(int steps) call.
    virtual void Progress(unsigned int steps) = 0;
  };

} // end namespace mitk

#endif
