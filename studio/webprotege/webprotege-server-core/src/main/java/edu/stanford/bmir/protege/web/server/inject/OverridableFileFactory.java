package edu.stanford.bmir.protege.web.server.inject;

import javax.annotation.Nonnull;
import javax.inject.Inject;
import java.io.File;

import static com.google.common.base.Preconditions.checkNotNull;

/**
 * Matthew Horridge
 * Stanford Center for Biomedical Informatics Research
 * 19 Mar 2017
 */
public class OverridableFileFactory {

    private File dataDirectory;

    @Inject
    public OverridableFileFactory(@Nonnull @DataDirectory File dataDirectory) {
        this.dataDirectory = checkNotNull(dataDirectory);
    }

    public OverridableFile getOverridableFile(@Nonnull String relativePathName) {
        return new OverridableFile(relativePathName, dataDirectory);
    }
}
