import {
  Box,
  Button,
  Grid,
  Heading,
  Image,
  Input,
  SimpleGrid,
  Text,
  VStack,
} from "@chakra-ui/react"
import { createFileRoute } from "@tanstack/react-router"
import { useState } from "react"

type EvaluationResults = {
  diagnosis: string
  confidence: number
  recommendations: string
}

export const Route = createFileRoute("/evaluate")({
  component: ImageEvaluation,
  beforeLoad: async () => {
    // TODO: handle unauthorized here
  },
})

function ImageEvaluation() {
  const [images, setImages] = useState<(string | null)[]>(Array(4).fill(null))
  const [isEvaluating, setIsEvaluating] = useState(false)
  const [results, setResults] = useState<EvaluationResults | null>(null)

  const handleImageUpload = (
    e: React.ChangeEvent<HTMLInputElement>,
    index: number,
  ) => {
    const file = e.target.files?.[0]
    if (file) {
      const reader = new FileReader()
      reader.onloadend = () => {
        const newImages = [...images]
        newImages[index] = reader.result as string
        setImages(newImages)
      }
      reader.readAsDataURL(file)
    }
  }

  const removeImage = (index: number) => {
    const newImages = [...images]
    newImages[index] = null
    setImages(newImages)
  }

  const handleEvaluate = async () => {
    setIsEvaluating(true)
    setTimeout(() => {
      setResults({
        diagnosis: "Normal",
        confidence: 0.95,
        recommendations: "Regular follow-up recommended",
      })
      setIsEvaluating(false)
    }, 2000)
  }

  const allImagesUploaded = images.every((img) => img !== null)

  return (
    <Box p={8}>
      <Grid templateColumns={{ base: "1fr", lg: "2fr 1fr" }} gap={8}>
        {/* Main Upload Section */}
        <Box>
          <Heading size="lg" mb={4}>
            Mammography Analyzer
          </Heading>
          <SimpleGrid columns={{ base: 1, md: 2 }} spacing={6}>
            <ImageUploadColumn
              title="Left"
              mloIndex={0}
              ccIndex={2}
              images={images}
              onUpload={handleImageUpload}
              onRemove={removeImage}
            />
            <ImageUploadColumn
              title="Right"
              mloIndex={1}
              ccIndex={3}
              images={images}
              onUpload={handleImageUpload}
              onRemove={removeImage}
            />
          </SimpleGrid>
        </Box>

        {/* Right Sidebar */}
        <VStack align="stretch" spacing={6}>
          <Box p={4} borderWidth="1px" rounded="xl" bg="white" boxShadow="sm">
            <Heading size="md" mb={2}>
              Ready to analyze
            </Heading>
            <Button
              colorPalette="brand"
              onClick={handleEvaluate}
              isLoading={isEvaluating}
              isDisabled={!allImagesUploaded}
              w="full"
            >
              Analyze
            </Button>
          </Box>

          <Box p={4} borderWidth="1px" rounded="xl" bg="white" boxShadow="sm">
            <Heading size="md" mb={2}>
              Patient Information
            </Heading>
            <Text color="gray.500">[Form or metadata goes here]</Text>
          </Box>

          {results && (
            <Box p={4} borderWidth="1px" rounded="xl" bg="white" boxShadow="sm">
              <Heading size="md" mb={2}>
                Analysis Results
              </Heading>
              <Text>
                <strong>Diagnosis:</strong> {results.diagnosis}
              </Text>
              <Text>
                <strong>Confidence:</strong>{" "}
                {(results.confidence * 100).toFixed(2)}%
              </Text>
              <Text>
                <strong>Recommendations:</strong> {results.recommendations}
              </Text>
            </Box>
          )}
        </VStack>
      </Grid>
    </Box>
  )
}

function ImageUploadColumn({
  title,
  mloIndex,
  ccIndex,
  images,
  onUpload,
  onRemove,
}: {
  title: string
  mloIndex: number
  ccIndex: number
  images: (string | null)[]
  onUpload: (e: React.ChangeEvent<HTMLInputElement>, i: number) => void
  onRemove: (i: number) => void
}) {
  return (
    <Box>
      <Heading size="md" mb={4} color="brand.700">
        {title}
      </Heading>
      <VStack align="stretch" spacing={4}>
        <ImageUploadBox
          label={`${title} Mediolateral Oblique (MLO)`}
          index={mloIndex}
          image={images[mloIndex]}
          onUpload={onUpload}
          onRemove={onRemove}
        />
        <ImageUploadBox
          label={`${title} Craniocaudal (CC)`}
          index={ccIndex}
          image={images[ccIndex]}
          onUpload={onUpload}
          onRemove={onRemove}
        />
      </VStack>
    </Box>
  )
}

function ImageUploadBox({
  label,
  index,
  image,
  onUpload,
  onRemove,
}: {
  label: string
  index: number
  image: string | null
  onUpload: (e: React.ChangeEvent<HTMLInputElement>, i: number) => void
  onRemove: (i: number) => void
}) {
  return (
    <Box>
      <Text mb={2} fontWeight="medium">
        {label}
      </Text>
      {image ? (
        <Box
          position="relative"
          border="1px solid"
          borderColor="gray.200"
          rounded="md"
          overflow="hidden"
        >
          <Image src={image} alt={label} w="full" />
          <Button
            position="absolute"
            top={1}
            right={1}
            size="xs"
            onClick={() => onRemove(index)}
            colorScheme="red"
          >
            ×
          </Button>
        </Box>
      ) : (
        <Box
          as="label"
          border="2px dashed"
          borderColor="brand.200"
          rounded="xl"
          h="150px"
          cursor="pointer"
          display="flex"
          alignItems="center"
          justifyContent="center"
          _hover={{ bg: "brand.50" }}
        >
          <Input
            type="file"
            accept="image/*"
            display="none"
            onChange={(e) => onUpload(e, index)}
          />
          <Text color="gray.400" fontSize="2xl">
            +
          </Text>
        </Box>
      )}
    </Box>
  )
}
