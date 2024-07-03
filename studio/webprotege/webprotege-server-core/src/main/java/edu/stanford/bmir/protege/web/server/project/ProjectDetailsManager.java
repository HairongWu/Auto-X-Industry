package edu.stanford.bmir.protege.web.server.project;

import edu.stanford.bmir.protege.web.shared.project.NewProjectSettings;
import edu.stanford.bmir.protege.web.shared.project.ProjectDetails;
import edu.stanford.bmir.protege.web.shared.project.ProjectId;
import edu.stanford.bmir.protege.web.shared.project.UnknownProjectException;
import edu.stanford.bmir.protege.web.shared.projectsettings.ProjectSettings;
import edu.stanford.bmir.protege.web.shared.user.UserId;

/**
 * Matthew Horridge
 * Stanford Center for Biomedical Informatics Research
 * 06/02/15
 */
public interface ProjectDetailsManager {
    /**
     * Creates a new project.
     * @param newProjectSettings The {@link NewProjectSettings} that describes the new project.  Not <code>null</code>.
     */
    void registerProject(ProjectId projectId, NewProjectSettings newProjectSettings);

    ProjectDetails getProjectDetails(ProjectId projectId) throws UnknownProjectException;

    boolean isExistingProject(ProjectId projectId);

    boolean isProjectOwner(UserId userId, ProjectId projectId);

    void setInTrash(ProjectId projectId, boolean b);

    ProjectSettings getProjectSettings(ProjectId projectId);

    void setProjectSettings(ProjectSettings projectSettings);


}
