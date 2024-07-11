/*============================================================================

The Medical Imaging Interaction Toolkit (MITK)

Copyright (c) German Cancer Research Center (DKFZ)
All rights reserved.

Use of this source code is governed by a 3-clause BSD license that can be
found in the LICENSE file.

============================================================================*/

#include "mitkNodePredicateAnd.h"

mitk::NodePredicateAnd::NodePredicateAnd() : NodePredicateCompositeBase()
{
}

mitk::NodePredicateAnd::NodePredicateAnd(const NodePredicateBase *p1, const NodePredicateBase *p2)
  : NodePredicateCompositeBase()
{
  this->AddPredicate(p1);
  this->AddPredicate(p2);
}

mitk::NodePredicateAnd::NodePredicateAnd(const NodePredicateBase *p1,
                                         const NodePredicateBase *p2,
                                         const NodePredicateBase *p3)
  : NodePredicateCompositeBase()
{
  this->AddPredicate(p1);
  this->AddPredicate(p2);
  this->AddPredicate(p3);
}

mitk::NodePredicateAnd::~NodePredicateAnd()
{
}

bool mitk::NodePredicateAnd::CheckNode(const mitk::DataNode *node) const
{
  if (m_ChildPredicates.empty())
    throw std::invalid_argument("NodePredicateAnd: no child predicates available");

  if (node == nullptr)
    throw std::invalid_argument("NodePredicateAnd: invalid node");

  // return the conjunction of the child predicate. If any predicate returns false, we return false too
  for (auto it = m_ChildPredicates.cbegin(); (it != m_ChildPredicates.cend()); ++it)
    if ((*it)->CheckNode(node) == false)
      return false; // if one element of the conjunction is false, the whole conjunction gets false
  return true;      // none of the childs was false, so return true
}
