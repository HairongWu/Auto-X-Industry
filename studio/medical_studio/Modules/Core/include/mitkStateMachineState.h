/*============================================================================

The Medical Imaging Interaction Toolkit (MITK)

Copyright (c) German Cancer Research Center (DKFZ)
All rights reserved.

Use of this source code is governed by a 3-clause BSD license that can be
found in the LICENSE file.

============================================================================*/

#ifndef mitkStateMachineState_h
#define mitkStateMachineState_h

#include "MitkCoreExports.h"
#include "mitkStateMachineTransition.h"
#include <itkLightObject.h>
#include <string>

namespace mitk
{
  /**
   * \class StateMachineState
   * Represents a state of a state machine pattern.
   * It holds transitions to other states (mitk::StateMachineTransition) and the mode of the current state, see
   * m_StateMode .
   */

  class MITKCORE_EXPORT StateMachineState : public itk::LightObject
  {
  public:
    mitkClassMacroItkParent(StateMachineState, itk::LightObject);
    mitkNewMacro2Param(Self, const std::string &, const std::string &);

    typedef std::vector<mitk::StateMachineState::Pointer> StateMap;
    typedef std::vector<StateMachineTransition::Pointer> TransitionVector;

    bool AddTransition(StateMachineTransition::Pointer transition);

    /**
    * @brief Return Transition which matches given event description.
    *
    * \deprecatedSince{2013_09} Use method GetTransitionList() instead.
    */
    DEPRECATED(StateMachineTransition::Pointer GetTransition(const std::string &eventClass,
                                                             const std::string &eventVariant));

    /**
    * @brief Return Transitions that match given event description.
    **/
    TransitionVector GetTransitionList(const std::string &eventClass, const std::string &eventVariant);

    /**
    * @brief Returns the name.
    **/
    std::string GetName() const;

    std::string GetMode() const;

    /**
    * @brief Searches dedicated States of all Transitions and sets *nextState of these Transitions.
    * Required for this is a List of all build States of that StateMachine (allStates). This way the StateMachine can be
    *build up.
    **/
    bool ConnectTransitions(StateMap *allStates);

  protected:
    StateMachineState(const std::string &name, const std::string &stateMode);
    ~StateMachineState() override;

  private:
    /**
    * @brief Name of this State.
    **/
    std::string m_Name;
    /**
     * State Modus, which determines the behavior of the dispatcher. A State can be in three different modes:
     * REGULAR - standard dispatcher behavior
     * GRAB_INPUT - all events are given to the statemachine in this modus, if they are not processed by this
     * statemachine
     * the events are dropped.
     * PREFER_INPUT - events are first given to this statemachine, and if not processed, offered to the other
     * statemachines.
     */
    std::string m_StateMode;

    /**
    * @brief map of transitions that lead from this state to the next state
    **/
    TransitionVector m_Transitions;
  };
} // namespace mitk
#endif
