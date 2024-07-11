/*============================================================================

The Medical Imaging Interaction Toolkit (MITK)

Copyright (c) German Cancer Research Center (DKFZ)
All rights reserved.

Use of this source code is governed by a 3-clause BSD license that can be
found in the LICENSE file.

============================================================================*/

#include "mitkMouseDoubleClickEvent.h"
#include "mitkException.h"

mitk::MouseDoubleClickEvent::MouseDoubleClickEvent(mitk::BaseRenderer *baseRenderer,
                                                   const mitk::Point2D &mousePosition,
                                                   MouseButtons buttonStates,
                                                   ModifierKeys modifiers,
                                                   MouseButtons eventButton)
  : InteractionPositionEvent(baseRenderer, mousePosition),
    m_EventButton(eventButton),
    m_ButtonStates(buttonStates),
    m_Modifiers(modifiers)
{
}

mitk::InteractionEvent::MouseButtons mitk::MouseDoubleClickEvent::GetEventButton() const
{
  return m_EventButton;
}

void mitk::MouseDoubleClickEvent::SetEventButton(MouseButtons buttons)
{
  m_EventButton = buttons;
}

mitk::InteractionEvent::ModifierKeys mitk::MouseDoubleClickEvent::GetModifiers() const
{
  return m_Modifiers;
}

mitk::InteractionEvent::MouseButtons mitk::MouseDoubleClickEvent::GetButtonStates() const
{
  return m_ButtonStates;
}

void mitk::MouseDoubleClickEvent::SetModifiers(ModifierKeys modifiers)
{
  m_Modifiers = modifiers;
}

void mitk::MouseDoubleClickEvent::SetButtonStates(MouseButtons buttons)
{
  m_ButtonStates = buttons;
}

mitk::MouseDoubleClickEvent::~MouseDoubleClickEvent()
{
}

bool mitk::MouseDoubleClickEvent::IsEqual(const mitk::InteractionEvent &interactionEvent) const
{
  const auto &mpe = static_cast<const mitk::MouseDoubleClickEvent &>(interactionEvent);
  return (this->GetEventButton() == mpe.GetEventButton() && this->GetModifiers() == mpe.GetModifiers() &&
          this->GetButtonStates() == mpe.GetButtonStates() && Superclass::IsEqual(interactionEvent));
}

bool mitk::MouseDoubleClickEvent::IsSuperClassOf(const InteractionEvent::Pointer &baseClass) const
{
  return (dynamic_cast<MouseDoubleClickEvent *>(baseClass.GetPointer()) != nullptr);
}
