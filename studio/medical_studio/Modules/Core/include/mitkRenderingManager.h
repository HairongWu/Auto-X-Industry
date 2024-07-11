/*============================================================================

The Medical Imaging Interaction Toolkit (MITK)

Copyright (c) German Cancer Research Center (DKFZ)
All rights reserved.

Use of this source code is governed by a 3-clause BSD license that can be
found in the LICENSE file.

============================================================================*/

#ifndef mitkRenderingManager_h
#define mitkRenderingManager_h

#include <MitkCoreExports.h>

#include <vtkCallbackCommand.h>

#include <itkObject.h>
#include <itkObjectFactory.h>

#include <mitkProperties.h>
#include <mitkPropertyList.h>
#include <mitkTimeGeometry.h>
#include <mitkAntiAliasing.h>

class vtkRenderWindow;
class vtkObject;

namespace mitk
{
  class RenderingManagerFactory;
  class BaseGeometry;
  class TimeNavigationController;
  class BaseRenderer;
  class DataStorage;

  /**
   * \brief Manager for coordinating the rendering process.
   *
   * RenderingManager is a central instance retrieving and executing
   * RenderWindow update requests. Its main purpose is to coordinate
   * distributed requests which cannot be aware of each other - lacking the
   * knowledge of whether they are really necessary or not. For example, two
   * objects might determine that a specific RenderWindow needs to be updated.
   * This would result in one unnecessary update, if both executed the update
   * on their own.
   *
   * The RenderingManager addresses this by letting each such object
   * <em>request</em> an update, and waiting for other objects to possibly
   * issue the same request. The actual update will then only be executed at a
   * well-defined point in the main event loop (this may be each time after
   * event processing is done).
   *
   * Convenience methods for updating all RenderWindows which have been
   * registered with the RenderingManager exist. If these methods are not
   * used, it is not required to register (add) RenderWindows prior to using
   * the RenderingManager.
   *
   * The methods #ForceImmediateUpdate() and #ForceImmediateUpdateAll() can
   * be used to force the RenderWindow update execution without any delay,
   * bypassing the request functionality.
   *
   * The interface of RenderingManager is platform independent. Platform
   * specific subclasses have to be implemented, though, to supply an
   * appropriate event issuing for controlling the update execution process.
   * See method documentation for a description of how this can be done.
   *
   * \sa TestingRenderingManager An "empty" RenderingManager implementation which
   * can be used in tests etc.
   *
   */
  class MITKCORE_EXPORT RenderingManager : public itk::Object
  {
  public:
    mitkClassMacroItkParent(RenderingManager, itk::Object);

    typedef std::vector<vtkRenderWindow *> RenderWindowVector;
    typedef std::vector<float> FloatVector;
    typedef std::vector<bool> BoolVector;

    typedef itk::SmartPointer<DataStorage> DataStoragePointer;

    enum RequestType
    {
      REQUEST_UPDATE_ALL = 0,
      REQUEST_UPDATE_2DWINDOWS,
      REQUEST_UPDATE_3DWINDOWS
    };

    static Pointer New();

    /** Set the object factory which produces the desired platform specific
     * RenderingManager singleton instance. */
    static void SetFactory(RenderingManagerFactory *factory);

    /** Get the object factory which produces the platform specific
     * RenderingManager instances. */
    static const RenderingManagerFactory *GetFactory();

    /** Returns true if a factory has already been set. */
    static bool HasFactory();

    /** Get the RenderingManager singleton instance. */
    static RenderingManager *GetInstance();

    /** Returns true if the singleton instance does already exist. */
    static bool IsInstantiated();

    /** Adds a RenderWindow. This is required if the methods #RequestUpdateAll
     * or #ForceImmediateUpdate are to be used. */
    void AddRenderWindow(vtkRenderWindow *renderWindow);

    /** Removes a RenderWindow. */
    void RemoveRenderWindow(vtkRenderWindow *renderWindow);

    /** Get a list of all registered RenderWindows */
    const RenderWindowVector &GetAllRegisteredRenderWindows();

    /** Requests an update for the specified RenderWindow, to be executed as
   * soon as the main loop is ready for rendering. */
    void RequestUpdate(vtkRenderWindow *renderWindow);

    /** Immediately executes an update of the specified RenderWindow. */
    void ForceImmediateUpdate(vtkRenderWindow *renderWindow);

    /** Requests all currently registered RenderWindows to be updated.
     * If only 2D or 3D windows should be updated, this can be specified
     * via the parameter requestType. */
    void RequestUpdateAll(RequestType type = REQUEST_UPDATE_ALL);

    /** Immediately executes an update of all registered RenderWindows.
     * If only 2D or 3D windows should be updated, this can be specified
     * via the parameter requestType. */
    void ForceImmediateUpdateAll(RequestType type = REQUEST_UPDATE_ALL);

    /**
    * @brief Initialize the render windows by the aggregated geometry of all objects that are held in
    *        the data storage.
    *
    * @param dataStorage       The data storage from which the bounding object can be retrieved
    */
    virtual void InitializeViewsByBoundingObjects(const DataStorage* dataStorage);

    /**
    * @brief Initialize the given render window by the aggregated geometry of all objects that are held in
    *        the data storage.
    *
    * @param renderWindow     The specified render window to update
    * @param dataStorage      The data storage from which the bounding object can be retrieved
    * @param resetCamera      If this parameter is set to true, the camera controller will be
    *                         set / fit to the center of the rendered image. If set to false, only the
    *                         the slice navigation controller is reset to the geometry without changing
    *                         the camera view / position.
    */

    virtual void InitializeViewByBoundingObjects(vtkRenderWindow* renderWindow,
                                                 const DataStorage* dataStorage,
                                                 bool resetCamera = true);

    /**
    * @brief Initialize the render windows specified by "type" to the given geometry.
    *
    * Throws an exception if bounding box has 0 extent due to exceeding double precision range.
    *
    * @param geometry              The geometry to be used to initialize / update a
    *                              render window's time and slice navigation controller
    * @param type                  The type of update request:
    *                                - REQUEST_UPDATE_ALL will initialize / update the
    *                                    time and slice navigation controller of all retrieved render windows
    *                                - REQUEST_UPDATE_2DWINDOWS will only initialize / update the
    *                                    time and slice navigation controller of 2D render windows
    *                                - REQUEST_UPDATE_3DWINDOWS will only initialize / update the
    *                                    time and slice navigation controller of 3D render windows
    * @param resetCamera           If this parameter is set to true, the camera controller will be
    *                              set / fit to the center of the rendered image. If set to false, only the
    *                              the slice navigation controller is reset to the geometry without changing
    *                              the camera view / position.
    */
    virtual bool InitializeViews(const BaseGeometry *geometry,
                                 RequestType type = REQUEST_UPDATE_ALL,
                                 bool resetCamera = true);

    /**
    * @brief Initialize the render windows specified by "type" to the given geometry.
    *
    * Throws an exception if bounding box has 0 extent due to exceeding double precision range.
    *
    * @param geometry              The geometry to be used to initialize / update a
    *                              render window's time- and slice navigation controller
    * @param type                  The type of update request:
    *                                - REQUEST_UPDATE_ALL will initialize / update the
    *                                    time- and slice navigation controller of all retrieved render windows
    *                                - REQUEST_UPDATE_2DWINDOWS will only initialize / update the
    *                                    time- and slice navigation controller of 2D render windows
    *                                - REQUEST_UPDATE_3DWINDOWS will only initialize / update the
    *                                    time- and slice navigation controller of 3D render windows
    * @param resetCamera           If this parameter is set to true, the camera controller will be
    *                              set / fit to the center of the rendered image. If set to false, only the
    *                              the slice navigation controller is reset to the geometry without changing
    *                              the camera view / position.
    */
    virtual bool InitializeViews(const TimeGeometry *geometry,
                                 RequestType type = REQUEST_UPDATE_ALL,
                                 bool resetCamera = true);

    /**
    * @brief Initialize the render windows specified by "type" to the default viewing direction
    *        without updating the geometry information.
    *
    * @param type                  The type of update request:
    *                                - REQUEST_UPDATE_ALL will initialize the
    *                                    slice navigation controller of all retrieved render windows
    *                                - REQUEST_UPDATE_2DWINDOWS will only initialize the
    *                                    slice navigation controller of 2D render windows
    *                                - REQUEST_UPDATE_3DWINDOWS will only initialize the
    *                                    slice navigation controller of 3D render windows
    */
    virtual bool InitializeViews(RequestType type = REQUEST_UPDATE_ALL);

    /**
    * @brief Initialize the specified render window to the given geometry.
    *
    * Throws an exception if bounding box has 0 extent due to exceeding double precision range.
    *
    * @param renderWindow          The specific render window to update
    * @param geometry              The geometry to be used to initialize / update the
    *                              render window's time- and slice navigation controller
    * @param resetCamera           If this parameter is set to true, the camera controller will be
    *                              set / fit to the center of the rendered image. If set to false, only the
    *                              the slice navigation controller is reset to the geometry without changing
    *                              the camera view / position.
    */
    virtual bool InitializeView(vtkRenderWindow *renderWindow,
                                const BaseGeometry *geometry,
                                bool resetCamera = true);

    /**
    * @brief Initialize the specified render window to the given geometry.
    *
    * Throws an exception if bounding box has 0 extent due to exceeding double precision range.
    *
    * @param renderWindow          The specific render window to update
    * @param geometry              The geometry to be used to initialize / update the
    *                              render window's time- and slice navigation controller
    * @param resetCamera           If this parameter is set to true, the camera controller will be
    *                              set / fit to the center of the rendered image. If set to false, only the
    *                              the slice navigation controller is reset to the geometry without changing
    *                              the camera view / position.
    */
    virtual bool InitializeView(vtkRenderWindow *renderWindow,
                                const TimeGeometry *geometry,
                                bool resetCamera = true);

    /**
    * @brief Initialize the specified render window to the default viewing direction
    *        without updating the geometry information.
    *
    * @param renderWindow          The specific render window to update
    */
    virtual bool InitializeView(vtkRenderWindow *renderWindow);

    /** Gets the (global) TimeNavigationController responsible for
     * time-slicing. */
    const TimeNavigationController* GetTimeNavigationController() const;

    /** Gets the (global) TimeNavigationController responsible for
     * time-slicing. */
    TimeNavigationController* GetTimeNavigationController();

    ~RenderingManager() override;

    /** Executes all pending requests. This method has to be called by the
     * system whenever a RenderingManager induced request event occurs in
     * the system pipeline (see concrete RenderingManager implementations). */
    virtual void ExecutePendingRequests();

    bool IsRendering() const;
    void AbortRendering();

    /** En-/Disable LOD increase globally. */
    itkSetMacro(LODIncreaseBlocked, bool);

    /** En-/Disable LOD increase globally. */
    itkGetMacro(LODIncreaseBlocked, bool);

    /** En-/Disable LOD increase globally. */
    itkBooleanMacro(LODIncreaseBlocked);

    /** En-/Disable LOD abort mechanism. */
    itkSetMacro(LODAbortMechanismEnabled, bool);

    /** En-/Disable LOD abort mechanism. */
    itkGetMacro(LODAbortMechanismEnabled, bool);

    /** En-/Disable LOD abort mechanism. */
    itkBooleanMacro(LODAbortMechanismEnabled);

    /** Force a sub-class to start a timer for a pending hires-rendering request */
    virtual void StartOrResetTimer(){};

    /** To be called by a sub-class from a timer callback */
    void ExecutePendingHighResRenderingRequest();

    virtual void DoStartRendering(){};
    virtual void DoMonitorRendering(){};
    virtual void DoFinishAbortRendering(){};

    int GetNextLOD(BaseRenderer *renderer);

    /** Set current LOD (nullptr means all renderers)*/
    void SetMaximumLOD(unsigned int max);

    void SetShading(bool state, unsigned int lod);
    bool GetShading(unsigned int lod);

    void SetClippingPlaneStatus(bool status);
    bool GetClippingPlaneStatus();

    void SetShadingValues(float ambient, float diffuse, float specular, float specpower);

    FloatVector &GetShadingValues();

    /** Returns a property list */
    PropertyList::Pointer GetPropertyList() const;

    /** Returns a property from m_PropertyList */
    BaseProperty *GetProperty(const char *propertyKey) const;

    /** Sets or adds (if not present) a property in m_PropertyList  */
    void SetProperty(const char *propertyKey, BaseProperty *propertyValue);

    /**
    * \brief Setter for internal DataStorage
    *
    * Sets the DataStorage that is used internally. This instance holds all DataNodes that are
    * rendered by the registered BaseRenderers.
    *
    * If this DataStorage is changed at runtime by calling SetDataStorage(),
    * all currently registered BaseRenderers are automatically given the correct instance.
    * When a new BaseRenderer is added, it is automatically initialized with the currently active DataStorage.
    */
    void SetDataStorage(DataStorage *storage);

    /**
    * \brief Getter for internal DataStorage
    *
    * Returns the DataStorage that is used internally. This instance holds all DataNodes that are
    * rendered by the registered BaseRenderers.
    */
    itkGetMacro(DataStorage, DataStorage*);
    itkGetConstMacro(DataStorage, DataStorage*);

    /**
     * @brief Sets a flag to the given renderwindow to indicated that it has the focus e.g. has been clicked recently.
     * @param focusWindow
     */
    void SetRenderWindowFocus(vtkRenderWindow *focusWindow);
    itkGetMacro(FocusedRenderWindow, vtkRenderWindow *);

    itkSetMacro(ConstrainedPanningZooming, bool);
    itkGetConstMacro(ConstrainedPanningZooming, bool);

    void SetAntiAliasing(AntiAliasing antiAliasing);
    itkGetConstMacro(AntiAliasing, AntiAliasing);

  protected:
    enum
    {
      RENDERING_INACTIVE = 0,
      RENDERING_REQUESTED,
      RENDERING_INPROGRESS
    };

    RenderingManager();

    /** Abstract method for generating a system specific event for rendering
     * request. This method is called whenever an update is requested */
    virtual void GenerateRenderingRequestEvent() = 0;

    virtual void InitializePropertyList();

    bool m_UpdatePending;

    typedef std::map<BaseRenderer *, unsigned int> RendererIntMap;
    typedef std::map<BaseRenderer *, bool> RendererBoolMap;

    RendererBoolMap m_RenderingAbortedMap;

    RendererIntMap m_NextLODMap;

    unsigned int m_MaxLOD;

    bool m_LODIncreaseBlocked;

    bool m_LODAbortMechanismEnabled;

    BoolVector m_ShadingEnabled;

    bool m_ClippingPlaneEnabled;

    FloatVector m_ShadingValues;

    static void RenderingStartCallback(vtkObject *caller, unsigned long eid, void *clientdata, void *calldata);
    static void RenderingProgressCallback(vtkObject *caller, unsigned long eid, void *clientdata, void *calldata);
    static void RenderingEndCallback(vtkObject *caller, unsigned long eid, void *clientdata, void *calldata);

    typedef std::map<vtkRenderWindow *, int> RenderWindowList;

    RenderWindowList m_RenderWindowList;
    RenderWindowVector m_AllRenderWindows;

    struct RenderWindowCallbacks
    {
      vtkCallbackCommand *commands[3u];
    };

    typedef std::map<vtkRenderWindow *, RenderWindowCallbacks> RenderWindowCallbacksList;

    RenderWindowCallbacksList m_RenderWindowCallbacksList;

    itk::SmartPointer<TimeNavigationController> m_TimeNavigationController;

    static RenderingManager::Pointer s_Instance;
    static RenderingManagerFactory *s_RenderingManagerFactory;

    PropertyList::Pointer m_PropertyList;

    DataStoragePointer m_DataStorage;

    bool m_ConstrainedPanningZooming;

  private:

    /**
    * @brief Initialize the specified renderer to the given geometry.
    *
    * @param baseRenderer            The specific renderer to update
    * @param geometry                The geometry to be used to initialize / update the
    *                                render window's slice navigation controller
    * @param boundingBoxInitialized  If this parameter is set to true, the slice navigation controller will be
    *                                initialized / updated with the given geometry. If set to false, the geometry
    *                                of the slice navigation controller is not updated.
    * @param mapperID                The mapper ID is used to define if the given renderer is a 2D or a 3D renderer.
    *                                In case of a 2D renderer and if "boundingBoxInitialized" is set to true (slice
    *                                navigation controller will be updated with a new geometry), the position of the
    *                                slice navigation controller is set to the center slice.
    * @param resetCamera             If this parameter is set to true, the camera controller will be
    *                                set / fit to the center of the rendered image. If set to false, only the
    *                                the slice navigation controller is reset to the geometry without changing
    *                                the camera view / position.
    */
    void InternalViewInitialization(BaseRenderer *baseRenderer,
                                    const TimeGeometry *geometry,
                                    bool boundingBoxInitialized,
                                    int mapperID,
                                    bool resetCamera);

    /**
    * @brief Extend the bounding box of the given geometry to make sure the bounding box has an extent bigger than
    *        zero in any direction.
    *
    * @param originalGeometry        The original geometry to be extended
    * @param modifiedGeometry        The modified geometry where the new bounds (extended bounding box) are used / set
    */
    bool ExtendGeometryForBoundingBox(const TimeGeometry* originalGeometry, TimeGeometry::Pointer& modifiedGeometry);

    vtkRenderWindow *m_FocusedRenderWindow;
    AntiAliasing m_AntiAliasing;
  };

#pragma GCC visibility push(default)

  itkEventMacroDeclaration(RenderingManagerEvent, itk::AnyEvent);
  itkEventMacroDeclaration(RenderingManagerViewsInitializedEvent, RenderingManagerEvent);

#pragma GCC visibility pop

  itkEventMacroDeclaration(FocusChangedEvent, itk::AnyEvent);

  /**
    * Generic RenderingManager implementation for "non-rendering-platform",
    * e.g. for tests. Its factory (TestingRenderingManagerFactory) is
    * automatically on start-up and is used by default if not other
    * RenderingManagerFactory is instantiated explicitly thereafter.
    * (see mitkRenderingManager.cpp)
    */
  class MITKCORE_EXPORT TestingRenderingManager : public RenderingManager
  {
  public:
    mitkClassMacro(TestingRenderingManager, RenderingManager);
    itkFactorylessNewMacro(Self);
    itkCloneMacro(Self);

  protected:
    
    void GenerateRenderingRequestEvent() override {};
  };

} // namespace mitk

#endif
