/*============================================================================

The Medical Imaging Interaction Toolkit (MITK)

Copyright (c) German Cancer Research Center (DKFZ)
All rights reserved.

Use of this source code is governed by a 3-clause BSD license that can be
found in the LICENSE file.

============================================================================*/
#ifndef mitkColorSequenceCycleH_h
#define mitkColorSequenceCycleH_h

#include "MitkDataTypesExtExports.h"
#include <mitkColorSequence.h>

namespace mitk
{
  /*!
    \brief Creates a list of around 36 different colors, where one is easily distinguished from the preceding one.

    The list of colors starts with a fully saturated, full valued red (Hue = 0 = 360).
    After that the sequence is generated like this:

    - first cycle through fully saturated colors (increase hue by 60)
    - then cycle through colors with halfed saturation (increase hue by 60)
    - then cycle through colors with halfed value (increase hue by 60)

    Finally repeat colors.

  */
  class MITKDATATYPESEXT_EXPORT ColorSequenceCycleH : public ColorSequence
  {
  public:
    ColorSequenceCycleH();

    ~ColorSequenceCycleH() override;

    /*!
    \brief Return another color
    */
    Color GetNextColor() override;

    /*!
    \brief Rewind to first color
    */
    void GoToBegin() override;

    /*!
    \brief Increase the used Hue value.
    This can be done by steps ( = steps * 60 increase of Hue )
    or absolute ( 0.0 < Hue < 360.0).
    Can also be used to decrease the Hue; Values < 0 are cropped to 0.
    Note: This does not change the other values, i.e. the color cycle.
    Therefor, the method can just be used to skip steps (i.e. colors) in a cycle.
    Use SetColorCycle if you want to change other values.
    */
    virtual void ChangeHueValueByCycleSteps(int steps);
    virtual void ChangeHueValueByAbsoluteNumber(float number);

    /*!
    \brief Set the color cycle.
    The color cycle has to be an integer value between 0 and 5
    (see class description for an explanation). Use this in combination with
    the hue value change to generate your dream colors...
    */
    virtual void SetColorCycle(unsigned short cycle);

  protected:
    float color_h; // current hue (0 .. 360)
    float color_s; // current saturation (0 .. 1)
    float color_v; // current value (0 .. 1)

    unsigned short color_cycle;
  };
}

#endif
