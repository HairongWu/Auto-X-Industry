package edu.stanford.bmir.protege.web.shared.issues;

import edu.stanford.bmir.protege.web.shared.project.HasProjectId;
import edu.stanford.bmir.protege.web.shared.annotations.GwtSerializationConstructor;
import edu.stanford.bmir.protege.web.shared.dispatch.ProjectAction;
import edu.stanford.bmir.protege.web.shared.project.ProjectId;

import javax.annotation.Nonnull;
import javax.inject.Inject;

import static com.google.common.base.Preconditions.checkNotNull;

/**
 * Matthew Horridge
 * Stanford Center for Biomedical Informatics Research
 * 7 Oct 2016
 */
public class AddEntityCommentAction implements ProjectAction<AddEntityCommentResult>, HasProjectId {

    private ProjectId projectId;

    private ThreadId threadId;

    private String comment;

    @Inject
    public AddEntityCommentAction(@Nonnull ProjectId projectId,
                                  @Nonnull ThreadId threadId,
                                  @Nonnull String comment) {
        this.projectId = checkNotNull(projectId);
        this.threadId = checkNotNull(threadId);
        this.comment = checkNotNull(comment);
    }

    public static AddEntityCommentAction addComment(@Nonnull ProjectId projectId,
                                                    @Nonnull ThreadId threadId,
                                                    @Nonnull String comment) {
        return new AddEntityCommentAction(projectId, threadId, comment);
    }

    @GwtSerializationConstructor
    private AddEntityCommentAction() {
    }

    @Nonnull
    public ProjectId getProjectId() {
        return projectId;
    }

    public ThreadId getThreadId() {
        return threadId;
    }

    public String getComment() {
        return comment;
    }
}
