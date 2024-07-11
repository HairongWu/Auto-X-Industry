/*============================================================================

The Medical Imaging Interaction Toolkit (MITK)

Copyright (c) German Cancer Research Center (DKFZ)
All rights reserved.

Use of this source code is governed by a 3-clause BSD license that can be
found in the LICENSE file.

============================================================================*/

#ifndef mitkDisplayActionEventHandler_h
#define mitkDisplayActionEventHandler_h

#include <MitkCoreExports.h>

// mitk core
#include "mitkDisplayActionEventBroadcast.h"
#include "mitkDisplayActionEvents.h"
#include "mitkStdFunctionCommand.h"

namespace mitk
{
  /**
  * @brief This class simplifies the process of adding an itkEventObject-itkCommand pair as an observer of a
  *        DisplayActionEventBroadcast instance.
  *        The 'SetObservableBroadcast'-function can be used to define the broadcast instance that should be observed.
  *        The 'ConnectDisplayActionEvent'-function can be used to add a an observer to the broadcast.
  *        Such an observer consists of a DisplayActionEvent (an itkEventObject) and a StdFunctionCommand (an itkCommand).
  *        The StdFunctionCommand is created inside the function by the given two std::functions.
  */
  class MITKCORE_EXPORT DisplayActionEventHandler
  {
  public:

    using OberserverTagType = unsigned long;

    virtual ~DisplayActionEventHandler();

    /**
    * @brief Sets the display action event broadcast class that should be observed.
    *     This class receives events from the given broadcast class and triggers the "corresponding functions" to perform the custom actions.
    *     "Corresponding functions" are std::functions inside commands that observe the specific display action event.
    *
    * @post If the same broadcast class was already set, nothing changed
    * @post If a different broadcast class was already set, the observing commands are removed as observer.
    *       Attention: All registered commands are removed from the list of observer.
    *
    * @param  observableBroadcast   The 'DisplayActionEventBroadcast'-class that should be observed.
    */
    void SetObservableBroadcast(DisplayActionEventBroadcast* observableBroadcast);

    /**
    * @brief Uses the given std::functions to customize a command:
    *     The display action event is used to define on which event the command should react.
    *     The display action event broadcast class member is then observed by the newly created command.
    *     A tag for the command is returned and stored in a member vector.
    *
    * @pre    The class' observable (the display action event broadcast) has to be set to connect display events.
    * @throw  mitk::Exception, if the class' observable is null.
    *
    * @param displayActionEvent   The 'DisplayActionEvent' on which the command should react.
    * @param actionFunction       A custom std::Function that will be executed if the command receives the correct event.
    * @param filterFunction       A custom std::Function that will be checked before the execution of the action function.
    *                           If the filter function is not specified, a default filter always returning 'true' will be used.
    *
    * @return   A tag to identify, receive or remove the newly created 'StdFunctionCommand'.
    */
    OberserverTagType ConnectDisplayActionEvent(const DisplayActionEvent& displayActionEvent,
      const StdFunctionCommand::ActionFunction& actionFunction,
      const StdFunctionCommand::FilterFunction& filterFunction = [](const itk::EventObject&) { return true; });

    /**
    * @brief Uses the given observer tag to remove the corresponding custom command as an observer of the observed
    *     display action event broadcast class.
    *     If the given tag is not contained in the member vector of observer tags, nothing happens.
    *
    * @pre    The class' observable (the display action event broadcast) has to be set to connect display events.
    * @throw  mitk::Exception, if the class' observable is null.
    *
    * @param observerTag   The tag to identify the 'StdFunctionCommand' observer.
    */
    void DisconnectObserver(OberserverTagType observerTag);

    const std::vector<OberserverTagType>& GetAllObserverTags() { return m_ObserverTags; }

    /**
    * @brief This function can be used by sub-classes to initialize a set of pre-defined
    *        DisplayActionEventFunctions and connect them to the observable broadcast member.
    *        In order to customize a sub-class behavior this function calls the virtual function
    *        InitActionsImpl.
    *        All currently connected display action events will be removed as observer from the broadcast instance.
    *
    * @pre    The class' observable (the display action event broadcast) has to be set.
    * @throw  mitk::Exception, if the class' observable is null.
    */
    void InitActions();

  protected:

    /**
    * @brief Sub-classes need to implement this function to define a customized behavior
    *        for default action pre-definition.
    */
    virtual void InitActionsImpl() { }

    WeakPointer<DisplayActionEventBroadcast> m_ObservableBroadcast;
    std::vector<OberserverTagType> m_ObserverTags;

  };

} // end namespace mitk

#endif
