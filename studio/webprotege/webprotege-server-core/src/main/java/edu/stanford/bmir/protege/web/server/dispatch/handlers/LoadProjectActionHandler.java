package edu.stanford.bmir.protege.web.server.dispatch.handlers;

import com.google.common.base.Stopwatch;
import edu.stanford.bmir.protege.web.server.access.AccessManager;
import edu.stanford.bmir.protege.web.server.dispatch.ApplicationActionHandler;
import edu.stanford.bmir.protege.web.server.dispatch.ExecutionContext;
import edu.stanford.bmir.protege.web.server.dispatch.RequestContext;
import edu.stanford.bmir.protege.web.server.dispatch.RequestValidator;
import edu.stanford.bmir.protege.web.server.dispatch.validators.ProjectPermissionValidator;
import edu.stanford.bmir.protege.web.server.project.ProjectDetailsManager;
import edu.stanford.bmir.protege.web.server.project.ProjectManager;
import edu.stanford.bmir.protege.web.server.user.UserActivityManager;
import edu.stanford.bmir.protege.web.server.util.MemoryMonitor;
import edu.stanford.bmir.protege.web.shared.project.LoadProjectAction;
import edu.stanford.bmir.protege.web.shared.project.LoadProjectResult;
import edu.stanford.bmir.protege.web.shared.project.ProjectDetails;
import edu.stanford.bmir.protege.web.shared.project.ProjectId;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import javax.annotation.Nonnull;
import javax.inject.Inject;
import java.util.concurrent.TimeUnit;

import static edu.stanford.bmir.protege.web.shared.access.BuiltInAction.VIEW_PROJECT;

/**
 * Author: Matthew Horridge<br>
 * Stanford University<br>
 * Bio-Medical Informatics Research Group<br>
 * Date: 05/04/2013
 */
public class LoadProjectActionHandler implements ApplicationActionHandler<LoadProjectAction, LoadProjectResult> {

    private static final Logger logger = LoggerFactory.getLogger(LoadProjectActionHandler.class);

    @Nonnull
    private final ProjectDetailsManager projectDetailsManager;

    @Nonnull
    private final ProjectManager projectManager;

    @Nonnull
    private final AccessManager accessManager;

    @Nonnull
    private final UserActivityManager userActivityManager;

    @Inject
    public LoadProjectActionHandler(@Nonnull ProjectDetailsManager projectDetailsManager,
                                    @Nonnull ProjectManager projectManager,
                                    @Nonnull AccessManager accessManager,
                                    @Nonnull UserActivityManager userActivityManager) {
        this.projectDetailsManager = projectDetailsManager;
        this.accessManager = accessManager;
        this.projectManager = projectManager;
        this.userActivityManager = userActivityManager;
    }

    @Nonnull
    @Override
    public Class<LoadProjectAction> getActionClass() {
        return LoadProjectAction.class;
    }

    @Nonnull
    @Override
    public RequestValidator getRequestValidator(@Nonnull LoadProjectAction action, @Nonnull RequestContext requestContext) {
        return new ProjectPermissionValidator(accessManager, action.getProjectId(), requestContext.getUserId(), VIEW_PROJECT
                .getActionId());
    }

    @Nonnull
    @Override
    public LoadProjectResult execute(@Nonnull final LoadProjectAction action, @Nonnull ExecutionContext executionContext) {
        Stopwatch stopwatch = Stopwatch.createStarted();
        logger.info("{} is being loaded due to request by {}",
                    action.getProjectId(),
                    executionContext.getUserId());
        projectManager.ensureProjectIsLoaded(action.getProjectId(), executionContext.getUserId());
        stopwatch.stop();
        logger.info("{} was loaded in {} ms due to request by {}",
                    action.getProjectId(),
                    stopwatch.elapsed(TimeUnit.MILLISECONDS),
                    executionContext.getUserId());
        MemoryMonitor memoryMonitor = new MemoryMonitor(logger);
        memoryMonitor.monitorMemoryUsage();
        memoryMonitor.logMemoryUsage();
        final ProjectId projectId = action.getProjectId();
        ProjectDetails projectDetails = projectDetailsManager.getProjectDetails(projectId);
        if (!executionContext.getUserId().isGuest()) {
            userActivityManager.addRecentProject(executionContext.getUserId(), action.getProjectId(), System.currentTimeMillis());
        }
        return LoadProjectResult.get(action.getProjectId(),
                                     executionContext.getUserId(),
                                     projectDetails);
    }
}
