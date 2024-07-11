/*============================================================================

The Medical Imaging Interaction Toolkit (MITK)

Copyright (c) German Cancer Research Center (DKFZ)
All rights reserved.

Use of this source code is governed by a 3-clause BSD license that can be
found in the LICENSE file.

============================================================================*/

#ifndef mitkInteractionTestHelper_h
#define mitkInteractionTestHelper_h

#include <mitkDataStorage.h>
#include <mitkDisplayActionEventBroadcast.h>
#include <mitkRenderWindow.h>
#include <mitkXML2EventParser.h>

#include <MitkTestingHelperExports.h>

class vtkRenderWindow;
class vtkRenderer;

namespace mitk
{
  /** @brief Creates everything needed to load and playback interaction events.
   *
    * The interaction is loaded from an xml file and the events are created. This file is
    * usually a recorded user interaction with the GUI. This can be done with InteractionEventRecorder
    * plugin. Also all necessary objects to handle interaction events are generated.
    * The user of this class is responsible to add the data object to interact with to the data storage
    * of InteractionTestHelper. And must also make sure that a proper data interactor is associated with the data
    * object.
    *
    * To test a PointSet interaction for instance make sure you have a PointSet node and a PointSetDataInteractor.
    * Then just add the node to the storage of the your InteractionTestHelper by calling
    * InteractionTestHelper::AddNodeToStorage.
    * Use InteractionTestHelper::PlaybackInteraction to execute. The result can afterwards be compared to a reference
    * object.
    *
    * Make sure to destroy the test helper instance after each test, since all render windows and its renderers have to
    * be unregistered.
    *
    * \sa XML2EventParser
    * \sa EventFactory
    * \sa EventRecorder
  */
  class MITKTESTINGHELPER_EXPORT InteractionTestHelper
  {
  public:
    /**
     * @brief InteractionTestHelper set up all neseccary objects by calling Initialize.
     * @param interactionXmlFilePath path to xml file containing events and configuration information for the render
     * windows.
     */
    InteractionTestHelper(const std::string &interactionXmlFilePath);

    // unregisters all render windows and its renderers.
    virtual ~InteractionTestHelper();

    /** @brief Returns the datastorage, in order to modify the data inside a rendering test.
      **/
    DataStorage::Pointer GetDataStorage();

    /**
       * @brief AddNodeToStorage Add a node to the datastorage and perform a reinit which is necessary for rendering.
       * @param node The data you want to add.
       */
    void AddNodeToStorage(DataNode::Pointer node);

    /**
     * @brief PlaybackInteraction playback loaded interaction by passing events to the dispatcher.
     */
    void PlaybackInteraction();

    /**
     * @brief SetTimeStep Sets timesteps of all SliceNavigationControllers to given timestep.
     * @param newTimeStep new timestep
     *
     * Does the same as using ImageNavigators Time slider. Use this if your data was modified in a timestep other than
     * 0.
     */
    void SetTimeStep(int newTimeStep);

    typedef std::vector<RenderWindow::Pointer> RenderWindowListType;

    const RenderWindowListType &GetRenderWindowList() { return m_RenderWindowList; }
    /**
     * @brief GetRenderWindowByName Get renderWindow by the name of its renderer.
     * @param name The name of the renderer of the desired renderWindow.
     * @return nullptr if not found.
     */
    RenderWindow *GetRenderWindowByName(const std::string &name);

    /**
     * @brief Get a renderWindow by its default view direction.
     * @param viewDirection
     * @return nullptr if not found.
     */
    RenderWindow *GetRenderWindowByDefaultViewDirection(AnatomicalPlane viewDirection);

    /**
     * @brief GetRenderWindow Get renderWindow at position 'index'.
     * @param index Position within the renderWindow list.
     * @return nullptr if index is out of bounds.
     */
    RenderWindow *GetRenderWindow(unsigned int index);

    /**
     * @brief AddDisplayPlaneSubTree
     *
     * Creates DisplayPlanes that are shown in a 3D RenderWindow.
     */
    void AddDisplayPlaneSubTree();

    void Set3dCameraSettings();

  protected:
    /**
    * @brief Initialize Internal method to initialize the renderwindow and set the datastorage.
    * @throws mitk::Exception if interaction xml file can not be loaded.
    */
    void Initialize(const std::string &interactionXmlFilePath);
    /**
    * @brief Initialize the interaction event observer / event state machine and register it as a service.
    */
    void InitializeDisplayActionEventHandling();
    /**
    * @brief LoadInteraction loads events from xml file.
    */
    void LoadInteraction();

    mitk::XML2EventParser::EventContainerType m_Events; // List with loaded interaction events

    std::string m_InteractionFilePath;

    RenderWindowListType m_RenderWindowList;
    DataStorage::Pointer m_DataStorage;
    DisplayActionEventBroadcast::Pointer m_DisplayActionEventBroadcast;

  };
}

#endif
