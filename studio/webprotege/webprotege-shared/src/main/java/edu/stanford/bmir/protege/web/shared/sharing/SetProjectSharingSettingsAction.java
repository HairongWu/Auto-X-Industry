package edu.stanford.bmir.protege.web.shared.sharing;

import com.google.common.base.Objects;
import edu.stanford.bmir.protege.web.shared.dispatch.ProjectAction;
import edu.stanford.bmir.protege.web.shared.project.ProjectId;

import javax.annotation.Nonnull;

import static com.google.common.base.MoreObjects.toStringHelper;
import static com.google.common.base.Preconditions.checkNotNull;

/**
 * Matthew Horridge
 * Stanford Center for Biomedical Informatics Research
 * 07/02/15
 */
public class SetProjectSharingSettingsAction implements ProjectAction<SetProjectSharingSettingsResult> {

    private ProjectSharingSettings projectSharingSettings;

    private SetProjectSharingSettingsAction() {
    }

    public SetProjectSharingSettingsAction(ProjectSharingSettings projectSharingSettings) {
        this.projectSharingSettings = checkNotNull(projectSharingSettings);
    }

    @Nonnull
    @Override
    public ProjectId getProjectId() {
        return projectSharingSettings.getProjectId();
    }

    public ProjectSharingSettings getProjectSharingSettings() {
        return projectSharingSettings;
    }

    @Override
    public int hashCode() {
        return Objects.hashCode(projectSharingSettings);
    }

    @Override
    public boolean equals(Object obj) {
        if (obj == this) {
            return true;
        }
        if (!(obj instanceof SetProjectSharingSettingsAction)) {
            return false;
        }
        SetProjectSharingSettingsAction other = (SetProjectSharingSettingsAction) obj;
        return this.projectSharingSettings.equals(other.projectSharingSettings);
    }


    @Override
    public String toString() {
        return toStringHelper("SetProjectSharingSettingsAction")
                .addValue(projectSharingSettings)
                .toString();
    }
}
