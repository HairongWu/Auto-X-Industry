/*============================================================================

The Medical Imaging Interaction Toolkit (MITK)

Copyright (c) German Cancer Research Center (DKFZ)
All rights reserved.

Use of this source code is governed by a 3-clause BSD license that can be
found in the LICENSE file.

============================================================================*/


#ifndef mitkUIDHelper_h
#define mitkUIDHelper_h

#include <string>

//MITK
#include "MitkMatchPointRegistrationExports.h"
#include "mitkMatchPointPropertyTags.h"

namespace mitk
{
  class DataNode;
  class BaseData;

  typedef std::string NodeUIDType;

  /** Gets the content of the property "node.uid". If it does not exist, the property will be added with a new UID.
  @pre Passed node is a valid pointer.*/
  NodeUIDType MITKMATCHPOINTREGISTRATION_EXPORT EnsureUID(mitk::DataNode* node);

  /** Helper that checks if the content of property "node.uid" equals the passed uid. If the property does not exist or node is invalid, return will be false.*/
  bool MITKMATCHPOINTREGISTRATION_EXPORT CheckUID(const mitk::DataNode* node, const NodeUIDType& uid);

  /** Gets the content of the property "data.uid". If it does not exist, the property will be added with a new UID.
  @pre Passed node is a valid pointer.*/
  NodeUIDType MITKMATCHPOINTREGISTRATION_EXPORT EnsureUID(mitk::BaseData* data);

  /** Helper that checks if the content of property "data.uid" equals the passed uid. If the property does not exist or node is invalid, return will be false.*/
  bool MITKMATCHPOINTREGISTRATION_EXPORT CheckUID(const mitk::BaseData* data, const NodeUIDType& uid);
}

#endif
