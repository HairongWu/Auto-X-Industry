/*============================================================================

The Medical Imaging Interaction Toolkit (MITK)

Copyright (c) German Cancer Research Center (DKFZ)
All rights reserved.

Use of this source code is governed by a 3-clause BSD license that can be
found in the LICENSE file.

============================================================================*/

#ifndef mitkIRenderWindowPart_h
#define mitkIRenderWindowPart_h

#include <QString>
#include <QStringList>
#include <QHash>
#include <QtPlugin>

#include <mitkBaseRenderer.h>
#include <mitkNumericTypes.h>
#include <mitkRenderingManager.h>

#include <org_mitk_gui_common_Export.h>

class QmitkRenderWindow;

namespace mitk {

struct IRenderingManager;
class TimeNavigationController;

/**
 * \ingroup org_mitk_gui_common
 *
 * \brief Interface for a MITK Workbench Part providing a render window.
 *
 * This interface allows generic access to Workbench parts which provide some
 * kind of render window. The interface is intended to be implemented by
 * subclasses of berry::IWorkbenchPart. Usually, the interface is implemented
 * by a Workbench editor.
 *
 * A IRenderWindowPart provides zero or more QmitkRenderWindow instances which can
 * be controlled via this interface. QmitkRenderWindow instances have an associated
 * \e id, which is implementation specific.
 * Additionally the defined values Axial, Sagittal, Coronal and Original from mitk::AnatomicalPlane
 * can be used to retrieve a specific QmitkRenderWindow.
 *
 * \see ILinkedRenderWindowPart
 * \see IRenderWindowPartListener
 * \see QmitkAbstractRenderEditor
 */
struct MITK_GUI_COMMON_PLUGIN IRenderWindowPart {

  static const QString DECORATION_BORDER; // = "border"
  static const QString DECORATION_LOGO; // = "logo"
  static const QString DECORATION_MENU; // = "menu"
  static const QString DECORATION_BACKGROUND; // = "background"
  static const QString DECORATION_CORNER_ANNOTATION; // = "corner annotation"

  virtual ~IRenderWindowPart();

  /**
   * Get the currently active (focused) render window.
   * Focus handling is implementation specific.
   *
   * \return The active QmitkRenderWindow instance; <code>nullptr</code>
   *         if no render window is active.
   */
  virtual QmitkRenderWindow* GetActiveQmitkRenderWindow() const = 0;

  /**
   * Get all render windows with their ids.
   *
   * \return A hash map mapping the render window id to the QmitkRenderWindow instance.
   */
  virtual QHash<QString,QmitkRenderWindow*> GetQmitkRenderWindows() const  = 0;

  /**
   * Get a render window with a specific id.
   *
   * \param id The render window id.
   * \return The QmitkRenderWindow instance for <code>id</code>
   */
  virtual QmitkRenderWindow* GetQmitkRenderWindow(const QString& id) const = 0;

  /**
  * Get a render window with a specific plane orientation.
  *
  * \param orientation The render window plane orientation.
  * \return The QmitkRenderWindow instance for <code>orientation</code>
  */
  virtual QmitkRenderWindow* GetQmitkRenderWindow(const mitk::AnatomicalPlane& orientation) const = 0;

  /**
   * Get the rendering manager used by this render window part.
   *
   * \return The current IRenderingManager instance or <code>nullptr</code>
   *         if no rendering manager is used.
   */
  virtual mitk::IRenderingManager* GetRenderingManager() const = 0;

  /**
   * Request an update of all render windows.
   *
   * \param requestType Specifies the type of render windows for which an update
   *        will be requested.
   */
  virtual void RequestUpdate(mitk::RenderingManager::RequestType requestType = mitk::RenderingManager::REQUEST_UPDATE_ALL) = 0;

  /**
   * Force an immediate update of all render windows.
   *
   * \param requestType Specifies the type of render windows for which an immediate update
   *        will be requested.
   */
  virtual void ForceImmediateUpdate(mitk::RenderingManager::RequestType requestType = mitk::RenderingManager::REQUEST_UPDATE_ALL) = 0;

   /**
   * @brief Initialize the render windows of this render window part to the given geometry.
    *
   * @param geometry      The geometry to be used to initialize / update a
    *                     render window's time and slice navigation controller.
   * @param resetCamera   If true, the camera and crosshair will be reset to the default view (centered, no zoom).
   *                      If false, the current crosshair position and the camera zoom will be stored and reset
   *                      after the reference geometry has been updated.
   */
  virtual void InitializeViews(const mitk::TimeGeometry* geometry, bool resetCamera) = 0;

  /**
  * @brief Define the reference geometry for interaction within a render window.
  *
  *        The concrete implementation is subclass-specific, no default implementation is provided here.
  *        An implementation can be found in 'QmitkAbstractMultiWidgetEditor' and will just
  *        forward the argument to the contained multi widget.
  *
  * @param referenceGeometry  The interaction reference geometry for the concrete multi widget.
  *                           For more details, see 'BaseRenderer::SetInteractionReferenceGeometry'.
  */
  virtual void SetInteractionReferenceGeometry(const mitk::TimeGeometry* referenceGeometry) = 0;

  /**
  * @brief Returns true if the render windows are coupled; false if not.
  *
  * Render windows are coupled if the slice navigation controller of the render windows
  * are connected which means that always the same geometry is used for the render windows.
  */
  virtual bool HasCoupledRenderWindows() const = 0;

  /**
   * Get the TimeNavigationController for controlling time positions.
   *
   * \return A TimeNavigationController if the render window supports this
   *         operation; otherwise returns <code>nullptr</code>.
   */
  virtual mitk::TimeNavigationController* GetTimeNavigationController() const = 0;

  /**
   * Get the selected position in the render window with id <code>id</code>
   * or in the active render window if <code>id</code> is an empty string.
   *
   * \param id The render window id.
   * \return The currently selected position in world coordinates.
   */
  virtual mitk::Point3D GetSelectedPosition(const QString& id = QString()) const = 0;

  /**
   * Set the selected position in the render window with id <code>id</code>
   * or in the active render window if <code>id</code> is nullptr.
   *
   * \param pos The position in world coordinates which should be selected.
   * \param id The render window id in which the selection should take place.
   */
  virtual void SetSelectedPosition(const mitk::Point3D& pos, const QString& id = QString()) = 0;

  /**
   * Get the time point selected in the render window with id <code>id</code>
   * or in the active render window if <code>id</code> is an empty string.
   *
   * \param id The render window id.
   * \return The currently selected position in world coordinates.
   */
  virtual TimePointType GetSelectedTimePoint(const QString& id = QString()) const = 0;

  /**
   * Enable \e decorations like colored borders, menu widgets, logos, text annotations, etc.
   *
   * Decorations are implementation specific. A set of standardized decoration names is listed
   * in GetDecorations().
   *
   * \param enable If <code>true</code> enable the decorations specified in <code>decorations</code>,
   *        otherwise disable them.
   * \param decorations A list of decoration names. If empty, all supported decorations are affected.
   *
   * \see GetDecorations()
   */
  virtual void EnableDecorations(bool enable, const QStringList& decorations = QStringList()) = 0;

  /**
   * Return if a specific decoration is enabled.
   *
   * \return <code>true</code> if the decoration is enabled, <code>false</code> if it is disabled
   *         or unknown.
   *
   * \see GetDecorations()
   */
  virtual bool IsDecorationEnabled(const QString& decoration) const = 0;

  /**
   * Get a list of supported decorations.
   *
   * The following decoration names are standardized and should not be used for other decoration types:
   * <ul>
   * <li>\e DECORATION_BORDER Any border decorations like colored rectangles, etc.
   * <li>\e DECORATION_MENU Menus associated with render windows
   * <li>\e DECORATION_BACKGROUND All kinds of backgrounds (patterns, gradients, etc.) except for solid colored backgrounds
   * <li>\e DECORATION_LOGO Any kind of logo overlayed on the rendered scene
   * </ul>
   *
   * \return A list of supported decoration names.
   */
  virtual QStringList GetDecorations() const = 0;
};

}

Q_DECLARE_INTERFACE(mitk::IRenderWindowPart, "org.mitk.ui.IRenderWindowPart")

#endif
