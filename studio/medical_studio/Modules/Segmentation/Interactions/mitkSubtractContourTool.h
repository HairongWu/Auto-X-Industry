/*============================================================================

The Medical Imaging Interaction Toolkit (MITK)

Copyright (c) German Cancer Research Center (DKFZ)
All rights reserved.

Use of this source code is governed by a 3-clause BSD license that can be
found in the LICENSE file.

============================================================================*/

#ifndef mitkSubtractContourTool_h
#define mitkSubtractContourTool_h

#include "mitkContourTool.h"
#include <MitkSegmentationExports.h>

namespace us
{
  class ModuleResource;
}

namespace mitk
{
  /**
    \brief Fill the inside of a contour with 1

    \sa ContourTool

    \ingroup Interaction
    \ingroup ToolManagerEtAl

    Fills a visible contour (from FeedbackContourTool) during mouse dragging. When the mouse button
    is released, SubtractContourTool tries to extract a slice from the working image and fill in
    the (filled) contour as a binary image. All inside pixels are set to 0.

    While holding the CTRL key, the contour changes color and the pixels on the inside would be
    filled with 1.


    \warning Only to be instantiated by mitk::ToolManager.

    $Author$
  */
  class MITKSEGMENTATION_EXPORT SubtractContourTool : public ContourTool
  {
  public:
    mitkClassMacro(SubtractContourTool, ContourTool);
    itkFactorylessNewMacro(Self);
    itkCloneMacro(Self);

      const char **GetXPM() const override;
    us::ModuleResource GetCursorIconResource() const override;
    us::ModuleResource GetIconResource() const override;

    const char *GetName() const override;

  protected:
    SubtractContourTool(); // purposely hidden
    ~SubtractContourTool() override;
  };

} // namespace

#endif
