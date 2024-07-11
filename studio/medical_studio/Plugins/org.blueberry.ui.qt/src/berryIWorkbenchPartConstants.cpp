/*============================================================================

The Medical Imaging Interaction Toolkit (MITK)

Copyright (c) German Cancer Research Center (DKFZ)
All rights reserved.

Use of this source code is governed by a 3-clause BSD license that can be
found in the LICENSE file.

============================================================================*/
#include "berryIWorkbenchPartConstants.h"

#include <QString>

namespace berry
{

const QString IWorkbenchPartConstants::INTEGER_PROPERTY = "org.blueberry.ui.integerproperty";

const int IWorkbenchPartConstants::PROP_TITLE = 0x001;
const int IWorkbenchPartConstants::PROP_DIRTY = 0x101;
const int IWorkbenchPartConstants::PROP_INPUT = 0x102;
const int IWorkbenchPartConstants::PROP_PART_NAME = 0x104;
const int IWorkbenchPartConstants::PROP_CONTENT_DESCRIPTION = 0x105;
const int IWorkbenchPartConstants::PROP_PREFERRED_SIZE = 0x303;

const int IWorkbenchPartConstants::PROP_OPENED = 0x211;
const int IWorkbenchPartConstants::PROP_CLOSED = 0x212;
const int IWorkbenchPartConstants::PROP_PINNED = 0x213;
const int IWorkbenchPartConstants::PROP_VISIBLE = 0x214;
// const int IWorkbenchPartConstants::PROP_ZOOMED = 0x215;
const int IWorkbenchPartConstants::PROP_ACTIVE_CHILD_CHANGED = 0x216;
// const int IWorkbenchPartConstants::PROP_MAXIMIZED = 0x217;

}
